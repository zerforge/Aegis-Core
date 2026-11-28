"""Tes make_eval_bundle dan logika murni eval_hf_bpb (tanpa torch/transformers)."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_hf_bpb as eb  # noqa: E402
import make_eval_bundle as mb  # noqa: E402


def w(p, texts, key="text"):
    p.write_text("".join(json.dumps({key: t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")


def test_make_windows():
    assert eb.make_windows(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert eb.make_windows(list(range(9)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]  # sisa 1 token dibuang
    with pytest.raises(ValueError):
        eb.make_windows([1, 2], 1)


def test_bits_per_byte_nilai_dan_penolakan():
    # 1000 nat, 100 dari 200 token terprediksi, 400 byte -> byte terprediksi 200
    assert eb.bits_per_byte(1000.0, 100, 200, 400) == pytest.approx(1000 / (math.log(2) * 200))
    with pytest.raises(ValueError):
        eb.bits_per_byte(1.0, 0, 10, 10)
    with pytest.raises(ValueError):
        eb.bits_per_byte(1.0, 11, 10, 10)


def test_score_corpus_dengan_model_palsu():
    docs = ["aaaa bbbb", "cc"]
    tokenize = lambda t: list(range(len(t)))  # noqa: E731  1 token per karakter
    calls = []

    def nll_fn(chunk):
        calls.append(len(chunk))
        return 2.0 * (len(chunk) - 1), len(chunk) - 1  # 2 nat per token terprediksi

    r = eb.score_corpus(nll_fn, tokenize, docs, window=4, has_bos=False)
    # dok1: 9 token -> [4,4] (sisa 1 dibuang); dok2: 2 token -> [2]
    assert calls == [4, 4, 2] and r["pred_tokens"] == 3 + 3 + 1 and r["tokens"] == 11
    assert r["bits_per_token"] == pytest.approx(2.0 / math.log(2))
    assert r["bytes"] == 11


def test_load_docs_memotong_dan_menolak(tmp_path):
    p = tmp_path / "a.jsonl"
    w(p, ["abcdef", "   ", "xyz"])
    assert eb.load_docs(p, 0, 4) == ["abcd", "xyz"]
    assert eb.load_docs(p, 1, 4) == ["abcd"]
    w(p, ["x"], key="lain")
    with pytest.raises(ValueError):
        eb.load_docs(p, 0, 4)


def test_bundle_deterministik_urutan_berkas_tidak_berpengaruh(tmp_path):
    texts = [f"dokumen nomor {i} " * 20 for i in range(30)]
    for name in mb.NAMES:
        w(tmp_path / f"{name}.jsonl", texts)
    w(tmp_path / "rev.jsonl", list(reversed(texts)))
    ins = {n: tmp_path / f"{n}.jsonl" for n in mb.NAMES}
    m1 = mb.build(ins, tmp_path / "o1", 10)
    ins2 = dict(ins, en=tmp_path / "rev.jsonl")
    m2 = mb.build(ins2, tmp_path / "o2", 10)
    assert m1["corpora"]["en"]["file_sha256"] == m2["corpora"]["en"]["file_sha256"]
    assert m1["corpora"]["en"]["docs"] == 10 and m1["corpora"]["cyber"]["docs"] == 30
    with pytest.raises(FileExistsError):
        mb.build(ins, tmp_path / "o1", 10)


def test_bundle_menolak_kunci_hilang_dan_melewati_dokumen_pendek(tmp_path):
    p = tmp_path / "x.jsonl"
    w(p, ["pendek", "panjang " * 50])
    texts, skipped = mb.read_texts(p)
    assert len(texts) == 1 and skipped == 1
    w(p, ["a"], key="konten")
    with pytest.raises(ValueError, match="konten"):
        mb.read_texts(p)


def test_dtype_kwarg_menurut_versi_transformers():
    assert eb.dtype_kwarg("4.57.1") == "torch_dtype"
    assert eb.dtype_kwarg("5.24.1") == "dtype"
    assert eb.dtype_kwarg("5.0.0.dev0") == "dtype"


def test_resolved_revision_tidak_menggagalkan_tanpa_cache():
    assert eb.resolved_revision("tidak/ada-model-ini", None) is None


def test_revision_record_sha_terpin_dipakai_apa_adanya():
    sha = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
    assert eb.revision_record("x/y", sha) == {"commit_hash": sha, "commit_hash_source": "pinned_arg",
                                              "commit_hash_error": None}
    assert eb.resolved_revision("x/y", sha) == sha


def test_revision_record_bukan_sha_mencatat_alasan_kegagalan():
    for rev in (None, "main", "a81bae4", "A81BAE42DB3975BE1671E27B9C9A56DA1A9F980F"):
        r = eb.revision_record("tidak/ada-model-ini", rev)
        assert r["commit_hash"] is None and r["commit_hash_source"] is None
        assert r["commit_hash_error"]  # tidak boleh diam-diam
