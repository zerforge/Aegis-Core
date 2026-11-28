"""Bits-per-byte model Hugging Face pada korpus tetap (v0.1.0, prototipe).

Untuk bake-off basis model (Fase 1): metrik sebanding antar tokenizer. Per dokumen: potong
ke --max-chars, tokenisasi, jendela tak-tumpang-tindih --window token, NLL token yang diprediksi.
Byte yang 'dimiliki' token terprediksi dihitung secara PROPORSIONAL: byte_dokumen x
(token_terprediksi / token_total). Ini aproksimasi; jangan bandingkan dengan angka dari metode lain.
Tanpa trust_remote_code. Gagal keras bila loss tidak terhingga/NaN (mis. overflow fp16).
Hanya angka yang disimpan; tidak ada teks.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

__version__ = "0.1.2"


def load_docs(path, max_docs, max_chars, key="text"):
    docs = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or not isinstance(obj.get(key), str):
            raise ValueError(f"{path}:{ln}: tidak ada kunci '{key}' bertipe str")
        text = obj[key][:max_chars]
        if text.strip():
            docs.append(text)
        if max_docs > 0 and len(docs) >= max_docs:
            break
    if not docs:
        raise ValueError(f"tidak ada dokumen di {path}")
    return docs


def make_windows(ids, size):
    """Potongan tak-tumpang-tindih sepanjang <= size; potongan < 2 token dibuang."""
    if size < 2:
        raise ValueError("window harus >= 2")
    return [c for c in (ids[i:i + size] for i in range(0, len(ids), size)) if len(c) >= 2]


def dtype_kwarg(transformers_version):
    """transformers 5.x menamai torch_dtype menjadi dtype; 4.x hanya mengenal torch_dtype."""
    return "dtype" if int(str(transformers_version).split(".")[0]) >= 5 else "torch_dtype"


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def revision_record(model_id, revision):
    """Catatan revisi model untuk JSON hasil. Bila --revision sudah berupa sha 40 heksa, itulah
    komit yang dimuat (sumber 'pinned_arg'). Selain itu coba cache lokal ('hf_cache'). Bila gagal,
    alasannya DICATAT (commit_hash_error), bukan ditelan diam-diam; evaluasi tidak digagalkan."""
    if revision and _SHA40.match(revision):
        return {"commit_hash": revision, "commit_hash_source": "pinned_arg", "commit_hash_error": None}
    try:
        from huggingface_hub import snapshot_download
        sha = Path(snapshot_download(model_id, revision=revision, local_files_only=True)).name
        return {"commit_hash": sha, "commit_hash_source": "hf_cache", "commit_hash_error": None}
    except Exception as e:  # noqa: BLE001 - metadata saja; jangan menggagalkan evaluasi
        return {"commit_hash": None, "commit_hash_source": None,
                "commit_hash_error": f"{type(e).__name__}: {e}"[:300]}


def resolved_revision(model_id, revision):
    """sha komit yang dipakai, atau None (lihat revision_record untuk alasannya)."""
    return revision_record(model_id, revision)["commit_hash"]


def bits_per_byte(nll_nats, n_pred, n_tokens, n_bytes):
    if min(n_pred, n_tokens, n_bytes) <= 0 or n_pred > n_tokens:
        raise ValueError("jumlah token/byte tidak valid")
    return nll_nats / (math.log(2) * n_bytes * n_pred / n_tokens)


def score_corpus(nll_fn, tokenize, docs, window, has_bos):
    """nll_fn(chunk_ids) -> (jumlah NLL nat, jumlah token terprediksi). Murni; mudah diuji."""
    tot_nll, n_pred, n_tok, n_bytes = 0.0, 0, 0, 0
    for text in docs:
        ids = tokenize(text)
        n_bytes += len(text.encode("utf-8"))
        n_tok += len(ids)
        for chunk in make_windows(ids, window):
            nll, k = nll_fn(chunk)
            tot_nll += nll
            n_pred += k
    return {"docs": len(docs), "bytes": n_bytes, "tokens": n_tok, "pred_tokens": n_pred,
            "nll_nats": tot_nll, "bits_per_token": tot_nll / math.log(2) / n_pred,
            "bits_per_byte": bits_per_byte(tot_nll, n_pred, n_tok, n_bytes)}


def run(a):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"device_map": "auto"}
    if a.load_4bit:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kw[dtype_kwarg(transformers.__version__)] = {
            "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model_id, revision=a.revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, revision=a.revision,
                                                 trust_remote_code=False, **kw)
    model.eval()
    bos = tok.bos_token_id
    dev = model.device

    def tokenize(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def nll_fn(chunk):
        ids = ([bos] if bos is not None else []) + chunk
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, :-1].float()
        loss = F.cross_entropy(logits, x[0, 1:], reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("loss tidak terhingga/NaN; coba --dtype lain atau --load-4bit")
        return float(loss), len(ids) - 1

    t0 = time.time()
    res = {}
    for spec in a.corpus:
        name, path = spec.split("=", 1)
        res[name] = score_corpus(nll_fn, tokenize, load_docs(path, a.max_docs, a.max_chars),
                                 a.window, bos is not None)
    out = {"script_version": __version__, "model_id": a.model_id, "revision_arg": a.revision,
           **revision_record(a.model_id, a.revision),
           "dtype": "4bit" if a.load_4bit else a.dtype, "window": a.window, "max_chars": a.max_chars,
           "max_docs": a.max_docs, "bos_prefix": bos is not None, "seconds": round(time.time() - t0, 1),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "peak_mem_gb": [round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                           for i in range(torch.cuda.device_count())],
           "corpora": res}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None, help="commit hash model (disarankan untuk reprodusibilitas)")
    ap.add_argument("--corpus", action="append", required=True, metavar="NAMA=JSONL")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = semua")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out", required=True, help="JSON hasil (menolak menimpa)")
    a = ap.parse_args(argv)
    if any("=" not in c for c in a.corpus) or a.max_chars < 1:
        raise SystemExit("GALAT: --corpus harus NAMA=JSONL dan --max-chars >= 1")
    if Path(a.out).exists():
        raise SystemExit(f"GALAT: sudah ada: {a.out}")
    out = run(a)
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {m: round(v[m], 4) for m in ("bits_per_byte", "bits_per_token")}
                      for k, v in out["corpora"].items()}, indent=1))
    print(f"waktu {out['seconds']} s; memori puncak GB: {out['peak_mem_gb']}", file=sys.stderr)


if __name__ == "__main__":
    main()
