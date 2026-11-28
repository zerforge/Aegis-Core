"""Tes logika murni eval_hf_mcq dengan data sintetis (tanpa torch/transformers, tanpa data berlisensi)."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_hf_mcq as em  # noqa: E402


def q(i, ans="A"):
    return {"question": f"pertanyaan sintetis {i}?", "answers": {k: f"opsi {k}{i}" for k in "ABCD"},
            "solution": ans}


def test_load_cybermetric_dan_baris_tidak_valid(tmp_path):
    p = tmp_path / "c.json"
    rows = [q(1, "B"), q(2, "Z"), dict(q(3, "C"), answers={"A": "x"})]
    p.write_text(json.dumps({"questions": rows}))
    items, bad = em.load_cybermetric(p)
    assert [i["answer"] for i in items] == ["B"] and bad == 2


def test_load_ctibench_normalisasi_huruf_kecil(tmp_path):
    p = tmp_path / "t.tsv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["URL", "Question", "Option A", "Option B", "Option C", "Option D", "Prompt", "GT"])
        w.writerow(["u", "apa 1?", "a", "b", "c", "d", "P", "b"])
        w.writerow(["u", "apa 2?", "a", "b", "c", "d", "P", "C"])
    items, bad = em.load_ctibench(p)
    assert [i["answer"] for i in items] == ["B", "C"] and bad == 0


def test_load_bench_format_tidak_dikenal(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a")
    with pytest.raises(ValueError):
        em.load_bench(p)


def make_items(n=12):
    return [em._item(i, f"soal {i}?", {k: f"o{k}{i}" for k in "ABCD"}, "ABCD"[i % 4]) for i in range(n)]


def test_split_few_shot_satu_per_huruf_dikeluarkan_dari_uji_dan_deterministik():
    items = make_items()
    shot, test = em.split_items(items, 4, 0)
    assert sorted(s["answer"] for s in shot) == list("ABCD")
    assert not {s["id"] for s in shot} & {t["id"] for t in test} and len(test) == 8
    assert em.split_items(list(reversed(items)), 4, 0) == (shot, test)
    assert len(em.split_items(items, 4, 3)[1]) == 3
    assert em.split_items(items, 0, 0)[0] == [] and len(em.split_items(items, 0, 0)[1]) == 12


def test_format_prompt_tepat():
    it = em._item(0, "Apa itu X?", {"A": "satu", "B": "dua", "C": "tiga", "D": "empat"}, "C")
    assert em.fmt_question(it) == "Question: Apa itu X?\nA. satu\nB. dua\nC. tiga\nD. empat\nAnswer:"
    p = em.build_prompt([it], it)
    assert p.startswith(em.HEADER) and "Answer: C\n\nQuestion: Apa itu X?" in p and p.endswith("Answer:")


def test_wilson_nilai_dan_penolakan():
    lo, hi = em.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert em.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        em.wilson(1, 0)


def test_summarize():
    s = em.summarize(list("AABC"), list("ABBC"))
    assert s["n"] == 4 and s["correct"] == 3 and s["accuracy"] == 0.75
    assert s["majority_letter"] == "A" and s["majority_baseline"] == 0.5
    assert s["gold_counts"] == {"A": 2, "B": 1, "C": 1} and s["pred_counts"] == {"A": 1, "B": 2, "C": 1}


def test_pred_records_tanpa_teks_dan_validasi_panjang():
    it = em._item(0, "rahasia?", {k: f"o{k}" for k in "ABCD"}, "B")
    rec = em.pred_records([em._key(it)[:16]], ["B"], ["C"])
    assert rec == [{"k": em._key(it)[:16], "gold": "B", "pred": "C"}]
    assert "rahasia" not in json.dumps(rec)
    with pytest.raises(ValueError):
        em.pred_records(["k"], ["A", "B"], ["A"])
