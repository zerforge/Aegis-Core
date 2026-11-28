"""Pemeriksaan struktur dan bias bentuk pada set MCQ buatan proyek (evals/mcq_*_v0.jsonl)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_hf_mcq as em  # noqa: E402

ID_SET = ROOT / "evals" / "mcq_id_v0.jsonl"
EN_SET = ROOT / "evals" / "mcq_en_parallel_v0.jsonl"
STATUS = {"unreviewed", "ok", "rejected"}


def rows(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.mark.parametrize("p", [ID_SET, EN_SET])
def test_set_valid_menurut_loader_dan_seimbang(p):
    r = rows(p)
    assert len(r) == 60
    items, bad = em.load_jsonl(p)
    assert bad == 0 and len(items) == len([x for x in r if x["review"] != "rejected"])
    assert len({x["id"] for x in r}) == 60 and len({x["question"] for x in r}) == 60
    assert {x["review"] for x in r} <= STATUS
    for x in r:
        texts = list(x["options"].values())
        assert list(x["options"]) == list("ABCD") and len(set(texts)) == 4 and all(t.strip() for t in texts)
        assert x["answer"] in "ABCD"
    counts = {k: sum(x["answer"] == k for x in r) for k in "ABCD"}
    assert max(counts.values()) - min(counts.values()) <= 1, counts


def test_set_paralel_id_dan_en_selaras():
    a, b = rows(ID_SET), rows(EN_SET)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert [x["answer"] for x in a] == [x["answer"] for x in b]
    assert [x["topic"] for x in a] == [x["topic"] for x in b]
    assert {x["lang"] for x in a} == {"id"} and {x["lang"] for x in b} == {"en"}


@pytest.mark.parametrize("p", [ID_SET, EN_SET])
def test_panjang_opsi_bukan_petunjuk(p):
    """Jawaban benar tidak boleh dominan sebagai opsi terpanjang, dan selisihnya tak boleh besar."""
    longest, worst = 0, -99
    for x in rows(p):
        L = {k: len(v) for k, v in x["options"].items()}
        c, others = L[x["answer"]], [v for k, v in L.items() if k != x["answer"]]
        longest += all(c >= v for v in others)
        worst = max(worst, c - max(others))
    assert longest <= 0.30 * 60, longest
    assert worst <= 6, worst


def test_review_rejected_tidak_pernah_dimuat_dan_only_reviewed_menyaring(tmp_path):
    base = rows(ID_SET)[0]
    recs = [dict(base, id="a", question="q1?", review="ok"), dict(base, id="b", question="q2?", review="unreviewed"),
            dict(base, id="c", question="q3?", review="rejected"), {k: v for k, v in dict(base, id="d", question="q4?").items() if k != "review"}]
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    assert [i["id"] for i in em.load_jsonl(p)[0]] == ["a", "b", "d"]
    assert [i["id"] for i in em.load_jsonl(p, only_reviewed=True)[0]] == ["a", "d"]
    q = tmp_path / "all_unreviewed.jsonl"
    q.write_text(json.dumps(dict(base, review="unreviewed")), encoding="utf-8")
    with pytest.raises(ValueError):
        em.load_bench(q, only_reviewed=True)
    with pytest.raises(ValueError):
        em.load_bench(tmp_path / "x.json", only_reviewed=True)


def test_pemilihan_contoh_dan_uji_sama_di_kedua_bahasa_dengan_by_id():
    a, _ = em.load_jsonl(ID_SET)
    b, _ = em.load_jsonl(EN_SET)
    sa, ta = em.split_items(a, 4, 0, by="id")
    sb, tb = em.split_items(b, 4, 0, by="id")
    assert [x["id"] for x in sa] == [x["id"] for x in sb] and len(sa) == 4
    assert [x["id"] for x in ta] == [x["id"] for x in tb] and len(ta) == 56
    # dengan by='question' urutan bergantung pada teks, jadi tidak dijamin sama
    sa2, _ = em.split_items(a, 4, 0)
    assert len(sa2) == 4
