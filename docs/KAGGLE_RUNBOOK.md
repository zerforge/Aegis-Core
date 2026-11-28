# Kaggle runbook: base-model bake-off, step 1 (bits per byte)

Status: DRAFT, not yet run. Purpose: compare candidate base models (docs/ROADMAP.md, Phase 1)
on domain text, general English and general Indonesian with one tokenizer-independent metric.
Rules: private notebook and private dataset; never put tokens or passwords in a notebook or
chat; record GPU hours used from the Kaggle quota counter (THREAT_MODEL R10).

## 1. On the local machine (WSL)

```bash
python scripts/make_eval_bundle.py \
  --cyber data/cyber_dataset/aegis_sec_v2_eval.jsonl \
  --en data/eval_indep_f1/en_eval.jsonl --id data/eval_indep_f1/id_eval.jsonl \
  --n-docs 200 --out-dir data/kaggle_bundle_v1
cat data/kaggle_bundle_v1/manifest.json
```

Copy the folder somewhere Windows can see (for example /mnt/c/Users/<name>/Downloads) and
upload its 4 files as a Kaggle Dataset, visibility PRIVATE (Create, New Dataset).

## 2. In a Kaggle notebook (Accelerator: GPU T4 x2; Internet: on)

```python
!pip -q install -U "transformers>=4.57" accelerate bitsandbytes
!git clone -q https://github.com/zerforge/Aegis-Core.git && git -C Aegis-Core log -1 --oneline
```

One cell per model (each run is a separate process, so memory is freed in between):

```python
D = "/kaggle/input/<dataset-slug>"
!python Aegis-Core/scripts/eval_hf_bpb.py --model-id allenai/Olmo-3-1025-7B --revision a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --corpus cyber={D}/cyber_eval.jsonl --corpus en={D}/en_eval.jsonl --corpus id={D}/id_eval.jsonl \
  --out /kaggle/working/bpb_olmo3_7b.json
```

```python
!python Aegis-Core/scripts/eval_hf_bpb.py --model-id Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --corpus cyber={D}/cyber_eval.jsonl --corpus en={D}/en_eval.jsonl --corpus id={D}/id_eval.jsonl \
  --out /kaggle/working/bpb_qwen3_8b.json
```

If a run fails with a non-finite loss (fp16 overflow), retry once with `--dtype bf16`
(T4 has no native bf16; it may be slow or fail) or `--load-4bit`, and record which was used:
results from different precisions are not directly comparable.

Revisions above are the ones used on 2026-11-28 (see docs/EXPERIMENTS.md). The result JSON
records the resolved revision from scripts/eval_hf_bpb.py 0.1.1 onward.

Paste the printed summary (bits per byte and per token per corpus, time, peak memory) and the
`commit_hash` field of each JSON into the chat. Lower bits per byte is better.

## What this does not show

Bits per byte measures how well a model predicts text. It does not measure cyber knowledge,
instruction following, refusal or safety. The byte attribution is proportional (see the script
docstring). Step 2 (below) adds multiple-choice cyber benchmarks; a project-written
Indonesian set is still to do.

# Step 2: multiple-choice cyber benchmarks (scripts/eval_hf_mcq.py)

Method: 4-shot prompt (one example per answer letter, chosen by sha256 order and excluded from
the test set), then the model's logits for " A" / " B" / " C" / " D" at the position after
`Answer:`; the highest letter is the prediction. Scores are comparable between models on this
method only, not with published numbers. Report accuracy with the Wilson 95% interval and the
majority-letter baseline (CTI-MCQ: always answering C gives about 37%, so 25% is not chance level
here). Licences: CTIBench is CC BY-NC-SA 4.0 and CyberMetric states no licence in its README:
internal evaluation only; do not redistribute them or commit them to the repo. The script prints
counts only, never question text (except `--show-first`, which prints one prompt to stderr, so
do not paste that output publicly).

## Kaggle cells (same notebook as step 1; Aegis-Core must be updated first)

```python
# 1. Update the repo copy in the notebook (or re-upload it) so scripts/eval_hf_mcq.py exists.
!ls Aegis-Core/scripts/eval_hf_mcq.py

# 2. Data. Use the same repository ids you used when you listed the files; if a download
#    fails, paste the error instead of changing ids by guesswork.
from huggingface_hub import hf_hub_download
import glob
CTI_REPO = "AI4Sec/cti-bench"        # as in your earlier list_repo_files call (verify)
cti = hf_hub_download(CTI_REPO, "cti-mcq.tsv", repo_type="dataset", local_dir="/kaggle/working/bench")
!git clone --depth 1 https://github.com/cybermetric/CyberMetric /kaggle/working/CyberMetric
cm = sorted(glob.glob("/kaggle/working/CyberMetric/CyberMetric-500*.json"))
print(cti, cm)
assert len(cm) == 1, cm
```

```python
# 3. Olmo 3 (use --show-first once to eyeball the exact prompt; it prints to stderr)
!python Aegis-Core/scripts/eval_hf_mcq.py --model-id allenai/Olmo-3-1025-7B --revision a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --bench cti_mcq={cti} --bench cybermetric500={cm[0]} --shots 4 --limit 500 --show-first \
  --out /kaggle/working/mcq_olmo3_7b.json
```

```python
# 4. Qwen3-8B-Base
!python Aegis-Core/scripts/eval_hf_mcq.py --model-id Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --bench cti_mcq={cti} --bench cybermetric500={cm[0]} --shots 4 --limit 500 \
  --out /kaggle/working/mcq_qwen3_8b.json
```

Notes:
- `--limit 500` takes the first 500 test questions in sha256 order (a fixed, content-based
  subset, identical for both models). Omit it to score all 2,500 CTI-MCQ questions (slower).
- The script stops with an error if " A".." D" is not one token in a tokenizer; tell me if so.
- Output files refuse to overwrite; use a new `--out` name for a rerun.
- Paste the printed summary (accuracy, interval, baseline, predicted-letter counts per
  benchmark), plus `commit_hash` and `skipped_too_long` from each JSON.
