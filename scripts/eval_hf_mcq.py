"""Benchmark pilihan ganda untuk model basis Hugging Face (v0.1.0, prototipe).

Few-shot + perbandingan logit huruf jawaban (A-D) pada token berikutnya setelah "Answer:".
Sumber: CTIBench cti-mcq.tsv (kolom Question, Option A-D, GT), CyberMetric *.json
({"questions":[{"question","answers":{A..D},"solution"}]}), atau JSONL proyek
({"id","question","options":{A..D},"answer"}). Skor HANYA sebanding antar model pada metode ini,
bukan dengan angka publik (yang memakai prompt berbeda).
Dilaporkan: akurasi + interval Wilson 95%, baseline kelas mayoritas, distribusi huruf gold dan
prediksi. Contoh few-shot (satu per huruf, urutan sha256) dikeluarkan dari pengujian.
Data benchmark berlisensi pihak ketiga (CTIBench: CC BY-NC-SA 4.0; CyberMetric: lisensi tidak
dinyatakan di README): hanya untuk evaluasi internal; keluaran hanya berisi hitungan, tanpa teks.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

__version__ = "0.2.0"
LETTERS = "ABCD"
HEADER = "The following are multiple-choice questions about cybersecurity. Answer with the letter of the correct option.\n\n"


def _item(i, question, options, answer):
    a = str(answer).strip().upper()
    if a not in LETTERS or set(options) != set(LETTERS) or not str(question).strip():
        return None
    return {"id": str(i), "question": str(question).strip(),
            "options": {k: str(options[k]).strip() for k in LETTERS}, "answer": a}


def load_cybermetric(path):
    items, bad = [], 0
    for i, q in enumerate(json.loads(Path(path).read_text(encoding="utf-8"))["questions"]):
        it = _item(i, q["question"], q["answers"], q["solution"])
        items.append(it) if it else None
        bad += it is None
    return items, bad


def load_ctibench(path):
    csv.field_size_limit(1 << 24)
    items, bad = [], 0
    with open(path, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            it = _item(i, r["Question"], {k: r[f"Option {k}"] for k in LETTERS}, r["GT"])
            items.append(it) if it else None
            bad += it is None
    return items, bad


def load_jsonl(path):
    items, bad = [], 0
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            r = json.loads(line)
            it = _item(r.get("id", i), r["question"], r["options"], r["answer"])
            items.append(it) if it else None
            bad += it is None
    return items, bad


def load_bench(path):
    suffix = Path(path).suffix
    loader = {".tsv": load_ctibench, ".json": load_cybermetric, ".jsonl": load_jsonl}.get(suffix)
    if loader is None:
        raise ValueError(f"format tidak dikenal: {suffix}")
    items, bad = loader(path)
    if not items:
        raise ValueError(f"tidak ada soal valid di {path}")
    return items, bad


def _key(item):
    return hashlib.sha256(item["question"].encode("utf-8")).hexdigest()


def split_items(items, shots, limit):
    """(contoh few-shot, soal uji). Contoh: satu per huruf gold (<= shots) menurut urutan sha256."""
    ordered = sorted(items, key=_key)
    shot, seen = [], set()
    for it in ordered:
        if it["answer"] not in seen and len(shot) < shots:
            shot.append(it)
            seen.add(it["answer"])
    ids = {id(s) for s in shot}
    test = [it for it in ordered if id(it) not in ids]
    return shot, (test[:limit] if limit > 0 else test)


def fmt_question(item):
    opts = "\n".join(f"{k}. {item['options'][k]}" for k in LETTERS)
    return f"Question: {item['question']}\n{opts}\nAnswer:"


def build_prompt(shot, item):
    parts = [HEADER] + [f"{fmt_question(s)} {s['answer']}\n\n" for s in shot] + [fmt_question(item)]
    return "".join(parts)


def wilson(k, n, z=1.96):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("k/n tidak valid")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(gold, pred):
    n = len(gold)
    k = sum(g == p for g, p in zip(gold, pred))
    lo, hi = wilson(k, n)
    maj = Counter(gold).most_common(1)[0]
    return {"n": n, "correct": k, "accuracy": k / n, "ci95": [round(lo, 4), round(hi, 4)],
            "majority_letter": maj[0], "majority_baseline": maj[1] / n,
            "gold_counts": dict(sorted(Counter(gold).items())),
            "pred_counts": dict(sorted(Counter(pred).items()))}


def pred_records(keys, gold, pred):
    """Catatan per soal tanpa teks: kunci = 16 heks pertama sha256 pertanyaan."""
    if not (len(keys) == len(gold) == len(pred)):
        raise ValueError("panjang keys/gold/pred berbeda")
    return [{"k": k, "gold": g, "pred": p} for k, g, p in zip(keys, gold, pred)]


def run(a):
    import time

    import torch
    import transformers
    from eval_hf_bpb import dtype_kwarg, revision_record
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
    letter_ids = []
    for L in LETTERS:
        ids = tok(" " + L, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"GALAT: ' {L}' bukan satu token pada tokenizer ini: {ids}")
        letter_ids.append(ids[0])
    t0, res = time.time(), {}
    for spec in a.bench:
        name, path = spec.split("=", 1)
        items, bad = load_bench(path)
        shot, test = split_items(items, a.shots, a.limit)
        if a.show_first:
            print(f"--- prompt pertama ({name}) ---\n{build_prompt(shot, test[0])}\n---", file=sys.stderr)
        gold, pred, keys, too_long = [], [], [], 0
        for it in test:
            ids = tok(build_prompt(shot, it))["input_ids"]
            if len(ids) > a.max_len:
                too_long += 1
                continue
            x = torch.tensor([ids], device=model.device)
            with torch.no_grad():
                logits = model(input_ids=x).logits[0, -1].float()
            pred.append(LETTERS[int(torch.argmax(logits[letter_ids]))])
            gold.append(it["answer"])
            keys.append(_key(it)[:16])
        r = summarize(gold, pred)
        if a.save_preds:
            r["items"] = pred_records(keys, gold, pred)
        r.update({"file": Path(path).name, "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  "invalid_rows": bad, "skipped_too_long": too_long, "shots": len(shot),
                  "fewshot_excluded_from_test": len(shot)})
        res[name] = r
    return {"script_version": __version__, "model_id": a.model_id, "revision_arg": a.revision,
            **revision_record(a.model_id, a.revision),
            "dtype": "4bit" if a.load_4bit else a.dtype, "shots": a.shots, "limit": a.limit,
            "seconds": round(time.time() - t0, 1), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "peak_mem_gb": [round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                            for i in range(torch.cuda.device_count())],
            "benchmarks": res}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--bench", action="append", required=True, metavar="NAMA=BERKAS")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="soal uji per benchmark (0 = semua)")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--show-first", action="store_true", help="cetak prompt pertama ke stderr")
    ap.add_argument("--save-preds", action="store_true",
                    help="simpan gold/prediksi per soal (kunci = 16 heks pertama sha256 pertanyaan; "
                         "tanpa teks) agar dua model bisa dibandingkan berpasangan")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if any("=" not in b for b in a.bench) or a.shots < 0 or a.shots > 4:
        raise SystemExit("GALAT: --bench NAMA=BERKAS dan 0 <= --shots <= 4")
    if Path(a.out).exists():
        raise SystemExit(f"GALAT: sudah ada: {a.out}")
    out = run(a)
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {m: v[m] for m in ("n", "accuracy", "ci95", "majority_baseline", "pred_counts")}
                      for k, v in out["benchmarks"].items()}, indent=1))
    print(f"waktu {out['seconds']} s; memori puncak GB: {out['peak_mem_gb']}", file=sys.stderr)


if __name__ == "__main__":
    main()
