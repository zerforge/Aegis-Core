# Experiments log

Results from small CPU-only experiments on Aegis-Nano (17,011,584 parameters).
All numbers are cross-entropy loss (nats per BPE-16K token) on the independent
evaluation set described in [DATA.md](../DATA.md). Every row is a single training
run per seed; no confidence intervals. Checkpoints, logs and raw result files
are not distributed (the `data/` directory is git-ignored).

## Common setup

6 layers, 6 heads, width 384, context 256, vocabulary 16,384 (tokenizer sha256
`a31e3dcf2c889f7b44ced454e560676767fcba0e871fc9bb9c4c3cb017cf4153`), batch 8,
AdamW (betas 0.9/0.95), weight decay 0.1, gradient clip 1.0, seed 1337 unless
stated. Training data: 46,651,978 tokens (data meta sha256
`248d6cf8785d6161e0336a1f7c3bf108e64eaab98ba5a2cfa6e13ccd4d386fdf`).
Python 3.14.4, torch 2.14.1+cpu, 4 cores, no GPU. Training is deterministic on
this machine: a rerun of the 2,000-step baseline (seed 1337) produced a
bit-identical checkpoint.

## Stage C: learning rate and dropout (2,000 steps, warmup 100)

Evaluation: window stride 16 (473,344 tokens).

| Cell | Peak LR | Dropout | Seed 1337 | Seed 2024 | Mean |
|---|---|---|---|---|---|
| C1 (control) | 1e-3 | 0.1 | 6.3567 | 6.3536 | 6.3552 |
| C3 | 3e-4 | 0.1 | 6.4033 | 6.4073 | 6.4053 |
| C4 | 1e-3 | 0.0 | 6.3179 | 6.3110 | 6.3145 |

Pre-registered rule: a difference counts only if it exceeds 0.02 nat and has the
same sign for both seeds. Seed-to-seed spread within a cell was 0.003 to 0.007.

- Lower LR (C3 vs C1): +0.050 nat, worse in both seeds. Hypothesis rejected at
  this training length.
- No dropout (C4 vs C1): -0.041 nat, better in both seeds. Supported at this
  training length.

## Longer check: 6,000 steps (warmup 200, seed 1337, one seed)

Evaluation: window stride 16.

| Cell | Dropout | Loss | Perplexity |
|---|---|---|---|
| C1 | 0.1 | 5.7004 | 299.0 |
| C4 | 0.0 | 5.5585 | 259.4 |

Difference -0.142 nat. The 8-batch training-log evaluation was lower for dropout
0 at every checkpoint (steps 999 to 5999).

## v4: full run without dropout (22,800 steps)

Identical to the v3 baseline (warmup 400, peak LR 1e-3 to 1e-4 cosine, seed
1337) except `--dropout 0.0`. Evaluation: window stride 4 (1,893,120 tokens).

| Model | Dropout | English | Indonesian | All | Perplexity |
|---|---|---|---|---|---|
| v3 (baseline) | 0.1 | 4.7396 | 4.8568 | 4.7882 | 120.1 |
| v4 | 0.0 | 4.6220 | 4.7263 | 4.6653 | 106.2 |

Difference -0.123 nat, one seed. Checkpoint sha256: v3
`5802ebe849b81ce31f1fb6fd623d1e25e199ebe323ca97d8444fcde760de3647`, v4
`ceec71d961dbd7c97bd55c34ab4c5156b52a21607329676a20d6379132a02522`. v4 took
about 17 h 43 min on CPU. Numbers measured at different strides are not
comparable with each other.

## Generation check: v3 vs v4

`scripts/eval_generation.py`, 10 fixed prompts (5 English, 5 Indonesian),
128 new tokens, 3 seeds for sampling strategies. "rep4" is the share of
generations that contain a repeated 4-gram.

| Strategy | rep4 en (v3 / v4) | rep4 id (v3 / v4) |
|---|---|---|
| greedy | 1.00 / 1.00 | 1.00 / 1.00 |
| top-k 40 | 0.80 / 0.87 | 0.60 / 0.87 |
| top-k 40, repetition penalty 1.2 | 0.33 / 0.60 | 0.27 / 0.20 |
| top-p 0.9 | 0.73 / 0.87 | 0.80 / 0.87 |
| top-p 0.9, repetition penalty 1.2 | 0.27 / 0.40 | 0.20 / 0.07 |

Each cell is about 15 generations (5 for greedy). Differences are within noise.
Reading samples: v4 text is locally more fluent but still not coherent.
Greedy decoding loops in both models, and the penalty reduces repetition at the
cost of malformed tokens.

## B1: 2x corpus, same compute (22,800 steps)

Same recipe as v4 (dropout 0, LR 1e-3, warmup 400, seed 1337, 22,800 steps),
trained on `data/tokens_2x`: 93,324,995 tokens (en 49,464,996, id 43,859,999),
about 2.0x the v4 corpus, from the same source files with more row groups. One
training document that was byte-identical to an independent-eval document was
removed before tokenization with `scripts/dedup_against_eval.py` (en: 42,095 ->
42,094 docs). Exact-text overlap with the independent eval was 0 for id.
Near-duplicates were not checked.

Independent eval, stride 4, same 1,893,120 tokens:

| | v4 | B1 | Diff |
|---|---|---|---|
| en | 4.6220 | 4.6075 | -0.0145 |
| id | 4.7263 | 4.7093 | -0.0170 |
| all | 4.6653 | 4.6497 | -0.0156 |

The pre-set criterion (more than 0.02 nat lower than v4) was **not met**.
Checkpoint sha256: `2716ec521c7f5efc1d5db2e2cd395abc5605fecd16e4f4dacc0fb0a7eba33d14`.

The run was interrupted and resumed with `--resume` from step 11,000. In the
log, 37 steps were written twice; 34 are identical and the largest difference
is 4.8e-7. No uninterrupted run exists, so bit-identical final weights are not
shown. The `eval_loss` in the training log uses the `tokens_2x` holdout and is
not comparable with v4.

Generation check (same settings as v3/v4), rep4 en / id:

| Strategy | B1 | v4 |
|---|---|---|
| greedy | 1.00 / 1.00 | 1.00 / 1.00 |
| top-k 40 | 0.80 / 0.80 | 0.87 / 0.87 |
| top-k 40, repetition penalty 1.2 | 0.27 / 0.20 | 0.60 / 0.20 |
| top-p 0.9 | 0.80 / 0.87 | 0.87 / 0.87 |
| top-p 0.9, repetition penalty 1.2 | 0.27 / 0.00 | 0.40 / 0.07 |

Greedy decoding still loops in 100% of generations. About 15 generations per
cell; differences are within noise. B1 text was not read, and rep4 does not
measure coherence.

Limits: one seed. The compute budget is fixed, so B1 sees about 0.5 epoch of
its corpus; it does not test longer training on more data. Source files are the
same as the v4 corpus, so this does not test data diversity in general. No
claim about language ability is made.

## v4 second seed (22,800 steps, seed 2024)

Same recipe and data as v4 (`data/tokens_full`); only `--seed 2024` and
`--out-dir data/runs_v4_s2024` differ. Independent eval, stride 4, same
1,893,120 tokens:

| | v4 seed 1337 | v4 seed 2024 | Diff | B1 (seed 1337) |
|---|---|---|---|---|
| en | 4.6220 | 4.6152 | -0.0068 | 4.6075 |
| id | 4.7263 | 4.7332 | +0.0069 | 4.7093 |
| all | 4.6653 | 4.6642 | -0.0011 | 4.6497 |

Checkpoint sha256: `185799894e961869c28dfa831008efb6e2dd4f3ba17209f25f8fbaba327e3bef`.

The run was interrupted and resumed with `--resume` from step 14,000. 30 steps
were written twice in the log and all are identical. The cause of the
interruption is unknown (B1 was also interrupted). The training-log `eval_loss`
(8 batches) was 4.6788 here versus 4.6384 for seed 1337; it depends on the seed
and is not used for comparison.

Reading: the spread between the two v4 seeds on the total is 0.0011 nat, much
smaller than the B1 difference (-0.0156 and -0.0145 against the two seeds). On
this evidence the B1 gain is probably not seed noise, but two seeds are not a
statistical estimate, B1 has one seed, and the pre-set 0.02 criterion is still
not met. Per-language differences between seeds (about 0.007) have opposite signs.

## Qualitative generation check: v4 vs B1

One prompt per language, seed 1337, strategies greedy, top-k 40, and top-p 0.9
with repetition penalty 1.2, first 300 characters of each sample. The prompt
texts are stored in `meta["prompts"]` of the result files (here "The history of the
city begins" and "Sejarah kota ini dimulai"). The samples were read by the AI
assistant, not by an independent human rater.

- Greedy loops in both models. en: sentence-level loops ("The city is ... of
  the city"). id: v4 collapses into lines of "-", B1 repeats one sentence.
- Top-k 40 repeats phrases within about 100 tokens in both models.
- The repetition penalty removes the loops but produces malformed tokens and
  invented names in both models.
- Short phrases are locally grammatical, but there is no coherence across
  sentences in either model.

Extra statistics over all greedy generations (n = 5 per cell):

| | v4 en | B1 en | v4 id | B1 id |
|---|---|---|---|---|
| Mostly punctuation | 0 | 0 | 1 | 1 |
| Mean position of first repeated 4-gram | 12.0 | 8.6 | 11.4 | 14.0 |

The differences point in opposite directions between languages and the cells
are tiny, so there is no signal. No evidence was found that B1 improves
generation over v4. The hypothesis that scale (17M parameters, 46-93M tokens)
is the main limit has not been tested. Spam or advertising drift in id output
was not checked systematically.

## Model B (GPU) generation evaluation: decoding strategies, en and id

Status: preliminary. One run per model. Source files: data/gen_eval/gen_eval_bgpu.json
(Model B, checkpoint ckpt_bgpu.pt, step 5700) and data/gen_eval/gen_eval_v4.json
(v4, step 22800). Both were produced by scripts/eval_generation.py (version
0.1.0) with the same tokenizer (data/tokenizer_cmp/v16384), the same 10 fixed
prompts (5 en, 5 id; identical prompt hash), seeds 1337+i for 3 seeds, and at
most 128 new tokens. Each sampling cell has 15 generations (5 prompts x 3
seeds, so not independent); each greedy cell has 5 (5 prompts, one seed).

Model B (identity as reported by the author; the layer, head and embedding
sizes, steps, batch size and LR schedule were later read from the checkpoint):
33,694,208 parameters (8 layers, 8 heads, 512 embedding dim), trained for 5700
steps on a training set of about 93.4M tokens (tokens_2x), on Kaggle GPU (2x
Tesla T4). v4 has about 17M parameters and was evaluated at step 22800, so the
two runs differ in size and in steps; the data and batch size differ as well
(v4: tokens_full, batch 8; Model B: tokens_2x, batch 64, read from the
checkpoint).

Metrics: d2/d3/d4 are distinct-n ratios; rep4% is the share of generations
containing a repeated 4-gram; firstrep is the mean token position of the first
repeated 4-gram, averaged only over the generations that repeat (so it is not
comparable across cells with different rep4%); tokens is the mean generated
length. Repetition is measured on token IDs.

| strategy | lang | d2 | d3 | d4 | rep4% | firstrep | tokens |
|---|---|---|---|---|---|---|---|
| greedy | en | 0.0945 | 0.1 | 0.1056 | 1.0 | 12.2 | 128.0 |
| greedy | id | 0.0913 | 0.1016 | 0.1104 | 1.0 | 10.8 | 128.0 |
| greedy_rep1.2 | en | 0.6173 | 0.7032 | 0.7616 | 1.0 | 44.2 | 128.0 |
| greedy_rep1.2 | id | 0.6126 | 0.6667 | 0.6992 | 1.0 | 30.6 | 128.0 |
| t1.0_pure | en | 0.9499 | 0.9852 | 0.9957 | 0.3333 | 86.0 | 114.5333 |
| t1.0_pure | id | 0.9608 | 0.9798 | 0.9864 | 0.4 | 83.1667 | 123.4 |
| t0.8_k40 | en | 0.646 | 0.7582 | 0.8152 | 0.9333 | 42.5 | 116.1333 |
| t0.8_k40 | id | 0.7837 | 0.873 | 0.9189 | 0.8 | 58.75 | 120.2667 |
| t0.8_k40_rep1.2 | en | 0.9648 | 0.9942 | 0.9995 | 0.0667 | 110.0 | 119.8 |
| t0.8_k40_rep1.2 | id | 0.9674 | 0.9917 | 0.9978 | 0.1333 | 56.5 | 110.6667 |
| t0.8_p0.9 | en | 0.6602 | 0.755 | 0.8082 | 0.9333 | 42.2143 | 122.0 |
| t0.8_p0.9 | id | 0.7243 | 0.782 | 0.8064 | 0.8 | 54.5833 | 114.2667 |
| t0.8_p0.9_rep1.2 | en | 0.978 | 0.9958 | 0.9984 | 0.0667 | 77.0 | 111.8667 |
| t0.8_p0.9_rep1.2 | id | 0.9658 | 0.9871 | 0.9924 | 0.3333 | 89.2 | 107.4667 |

Comparison with v4 (rep4%, same protocol, from the two result files):

| Strategy | rep4 en (v4 / Model B) | rep4 id (v4 / Model B) |
|---|---|---|
| greedy | 1.00 / 1.00 | 1.00 / 1.00 |
| top-k 40 | 0.87 / 0.93 | 0.87 / 0.80 |
| top-k 40, repetition penalty 1.2 | 0.60 / 0.07 | 0.20 / 0.13 |
| top-p 0.9 | 0.87 / 0.93 | 0.87 / 0.80 |
| top-p 0.9, repetition penalty 1.2 | 0.40 / 0.07 | 0.07 / 0.33 |

Share of generations that stopped on EOS (penalized strategies), v4 / Model B:

| Strategy | en | id |
|---|---|---|
| top-k 40, repetition penalty 1.2 | 0.33 / 0.20 | 0.53 / 0.40 |
| top-p 0.9, repetition penalty 1.2 | 0.20 / 0.20 | 0.60 / 0.53 |

Held-out language-model loss, same eval set for all three (data/tokens_eval_indep,
1,893,120 tokens, block 256, window stride 4, identical eval-set hash; from
data/eval_results/results_{v4,b1,bgpu}_s4.json):

| model | params | step | loss | perplexity | loss en | loss id |
|---|---|---|---|---|---|---|
| v4 | 17.0M | 22800 | 4.665 | 106.2 | 4.622 | 4.726 |
| B1 | 17.0M | 22800 | 4.650 | 104.6 | 4.607 | 4.709 |
| Model B | 33.7M | 5700 | 3.968 | 52.9 | 3.959 | 3.980 |

Model B's loss is about 0.70 nats lower than v4 (perplexity about halved), in
both languages. The v4 vs B1 difference is 0.016, so the gap is much larger
than the differences seen between the 17M runs. One run, one seed. Size,
tokens seen and data changed together, so the cause is not isolated: v4 and B1
saw about 46.7M tokens (22800 steps x batch 8 x block 256), and Model B saw
about 93.4M tokens (5700 steps x batch 64 x block 256).

Findings:

- Without the repetition penalty, Model B and v4 are about the same: greedy is
  1.00 for both, and top-k 40 and top-p 0.9 differ by one generation of 15.
- With the penalty, English rep4 is lower for Model B in both strategies
  (0.60 to 0.07 and 0.40 to 0.07, i.e. 9 to 1 and 6 to 1 generations of 15).
  Indonesian is mixed: 0.20 to 0.13 under top-k 40, but 0.07 to 0.33 under
  top-p 0.9 (1 to 5 of 15).
- The English gap is the only large difference. Generations share 5 prompts, so
  the effective sample is smaller than 15 and significance is not established.
- Early stopping does not explain the English gap: Model B stopped on EOS no
  more often than v4, so its generations were not shorter.
- Under the penalty, distinct-n is about 0.99 or higher (d4 0.998 to 0.9995).
  That alone does not indicate coherence. The v4 vs B1 check above found that
  the penalty produces malformed tokens and invented names.
- Samples were read (7 generations, seed 1337: penalized top-k 40 and top-p 0.9
  on en prompts 0 and 1, penalized id prompt 0 for both strategies, and one
  unpenalized top-k 40 en). Without the penalty, the en sample collapses into a
  loop ("Vell-Vell-Vell-..."). With the penalty there are no loops, and the
  first sentence is usually fluent, but later text drifts between topics and
  contains contradictory or invented content (for example "dates back to the
  19th century" followed by "20,000 years after its destruction"; the names
  "Arrangel Röscherberg" and "Power Floor Stewart, NSW"; Europe, Japan and
  Rasulullah SAW in one id paragraph), and table or list fragments from the
  training data ("|Developmentary||Starting with Buddhism|"). This is a small
  hand-read sample, not a rating.
- Greedy firstrep (12.2 en, 10.8 id) is close to the v4 greedy values reported
  above (12.0 en, 11.4 id), so greedy looping looks unchanged. firstrep 110.0
  at rep4 0.0667 (top-k 40 with penalty, en) rests on a single generation.
- Pure sampling (t1.0_pure) gives rep4 0.33 en and 0.40 id with d4 about 0.99.
  It has no v4 counterpart in the comparison above.

What this does not show:

- Coherence or text quality of Model B. rep4% dropped under the penalty, but
  the samples read show no evidence that text quality improved.
- That Model B improves on v4 without a repetition penalty.
- That scale or training length is the cause. Model B has about twice the
  parameters of v4 but 5700 steps against 22800, and batch 64 against 8
  (about 93.4M tokens seen against 46.7M), on different data, so size, steps,
  batch and data are all different.
- Variation beyond the fixed prompt set and the seeds used.

Next: add a second seed for Model B, and add prompts and seeds to the
generation check. Model B's batch size and tokens seen were read from its
checkpoint (see the 9x section below).

## What these results do not show

- Lower loss did not produce measurably less repetition or coherent text.
  No claim about language ability is made.
- Findings come from one seed at 6,000 steps, two seeds at 2,000 steps, and
  two seeds at 22,800 steps for v4 only (B1 has one). Only the LR values 3e-4 and 1e-3, and dropout 0 and 0.1, were tried.
- The model sees about one epoch-equivalent of data. Dropout 0 was not tested
  with repeated data.
- No task benchmark, factuality, safety or memorization evaluation has been run.

## Reproduce v4

```bash
python scripts/train.py --data-dir data/tokens_full --out-dir data/runs_v4 \
  --seed 1337 --batch-size 8 --block-size 256 --n-layer 6 --n-head 6 --n-embd 384 \
  --dropout 0.0 --max-steps 22800 --lr 1e-3 --min-lr 1e-4 --warmup 400 \
  --weight-decay 0.1 --grad-clip 1.0 --log-interval 10 --ckpt-interval 500 \
  --eval-interval 2000 --eval-batches 8
python scripts/evaluate_lm.py --ckpt v4=data/runs_v4/ckpt.pt \
  --eval-set indep=data/tokens_eval_indep:4 --out data/eval_results/results_v4_s4.json
```

## Reproduce B1

```bash
python scripts/sample_corpus.py --profile full2x --run
python scripts/dedup_against_eval.py \
  --train-files data/sample_2x/en_train.jsonl data/sample_2x/id_train.jsonl \
  --eval-files data/eval_indep/en_eval.jsonl data/eval_indep/id_eval.jsonl \
  --out-dir data/sample_2x_dd
python scripts/prepare_tokens.py --split train \
  --files data/sample_2x_dd/en_train.jsonl data/sample_2x_dd/id_train.jsonl \
  --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_2x
python scripts/prepare_tokens.py --split eval \
  --files data/sample_2x/en_eval.jsonl data/sample_2x/id_eval.jsonl \
  --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_2x
python scripts/train.py --data-dir data/tokens_2x --out-dir data/runs_b1 \
  --seed 1337 --batch-size 8 --block-size 256 --n-layer 6 --n-head 6 --n-embd 384 \
  --dropout 0.0 --max-steps 22800 --lr 1e-3 --min-lr 1e-4 --warmup 400 \
  --weight-decay 0.1 --grad-clip 1.0 --log-interval 10 --ckpt-interval 500 \
  --eval-interval 2000 --eval-batches 8
python scripts/evaluate_lm.py --ckpt b1=data/runs_b1/ckpt.pt \
  --eval-set indep=data/tokens_eval_indep:4 --out data/eval_results/results_b1_s4.json
```

## Model B learning-rate sweep (1,500 steps, GPU)

Model B (8 layers, 8 heads, 512 embedding dim) on `tokens_2x`, batch 64, block
256, 1,500 steps, warmup 100, seed 1337, dropout 0, fp16, minimum LR = LR / 10.
Four LRs, two at a time on the two T4 GPUs (one run per GPU, selected with
`CUDA_VISIBLE_DEVICES`). `eval_loss` is the loss on the `tokens_2x` holdout (32
batches), so it is comparable only inside this sweep, not with the evaluation
numbers elsewhere in this file.

| Step | LR 3e-4 | LR 6e-4 | LR 1e-3 | LR 2e-3 |
|---|---|---|---|---|
| 249 | 6.4535 | 6.3039 | 6.2412 | 6.2826 |
| 499 | 5.9788 | 5.7641 | 5.6540 | 5.7358 |
| 749 | 5.6432 | 5.3936 | 5.2656 | 5.3946 |
| 999 | 5.4385 | 5.1640 | 5.0135 | 5.1573 |
| 1249 | 5.3216 | 5.0243 | 4.8419 | 4.9880 |
| 1499 | 5.2594 | 4.9482 | 4.7448 | 4.8889 |

Rule fixed before the results: pick the lowest final `eval_loss`; if two are
within 0.02, pick the lower LR. LR 1e-3 had the lowest `eval_loss` at every
evaluation from step 249 on, and was 0.144 below the next best (2e-3) at step
1,499. It is not at the edge of the grid. No logged step (every 10th) had a
non-finite gradient norm in any run.

Limits: short runs tend to favour larger LRs; one seed; the grid spacing is about
a factor of 2, so the optimum may lie anywhere between 6e-4 and 2e-3.

The two waves took 448 s and 463 s for 1,500 steps x 16,384 tokens per run, i.e.
at least about 54,900 and 53,100 tokens per second per T4. This is a lower bound,
because the wall time includes startup and six evaluations.

## GPU smoke test (Kaggle, 100 steps)

`scripts/train.py` v0.2.0 (commit `9e76e3f`), torch 2.11.0+cu130, 2 x Tesla T4
(15,360 MiB each; only one is used), the 17M configuration (6 layers, 6 heads, 384)
on `tokens_full`, batch 64, block 256, fp16, notebook private (reported by
the author). Data hashes and the
tokenizer hash were checked inside the notebook.

- 100 steps run as 0-50 (`--stop-after 50`) and 50-100 (`--resume`); the two
  segments took 18.3 s and 15.1 s of wall time (startup included), i.e. at least
  about 44,900 and 54,300 tokens per second.
- Training loss fell from 9.13 (step 4) to 7.32 (step 99); `eval_loss` 7.49 at
  step 49 and 7.28 at step 99. No logged step had a non-finite gradient norm.
- The checkpoint had `amp` = fp16 and a non-empty scaler state; the scaler's
  growth tracker was 100 at step 100 across the resume, which is consistent with
  the scaler state having been restored.
- `--time-budget-hours` stopped a run cleanly at step 36 and the resume completed
  it to step 100.
- The checkpoint was opened on CPU with `map_location="cpu"`.

This tests the mechanics only; a loss of 7.3 after 100 steps says nothing about
quality.

## Model B on the 9x corpus (23,700 steps)

Same model and settings as the Model B control (8 layers, 8 heads, 512 embedding
dim, batch 64, block 256, LR 1e-3 to 1e-4, weight decay 0.1, clip 1.0, dropout 0,
seed 1337, fp16) except as listed below. Trained on Kaggle with Save & Run All (one
T4 used), data `data/tokens_9x_clean` (see `DATA.md`): 388,442,420 tokens
(en 221,440,917, id 167,001,503).

Checkpoint `data/ckpt_b9x.pt`, sha256
`5ef282664c70fa6e971f250f78e0c970205350987a464740da50a590ad6c5c66`. Checked on
the downloaded file: step 23700, `amp` fp16, script version 0.2.0, the data meta
hash in the checkpoint equals the sha256 of `meta_train.json`
(`2f84aca435dfc7a12ffb90c4ca903a8558e2d2cc2d135415e7b96b74035aa5ee`), the
tokenizer hash equals `a31e3dcf2c889f7b44ced454e560676767fcba0e871fc9bb9c4c3cb017cf4153`,
and the schedule in the checkpoint matches the command that was run. The hash in
`results_b9x_s4.json` equals the hash above. There is no hash from the Kaggle
side for this run (seed 1337).

Independent eval, stride 4, same 1,893,120 tokens as for the other models:

| model | params | tokens seen | loss | perplexity | loss en | loss id |
|---|---|---|---|---|---|---|
| v4 | 17.0M | 46.7M | 4.665 | 106.2 | 4.622 | 4.726 |
| B1 | 17.0M | 46.7M | 4.650 | 104.6 | 4.607 | 4.709 |
| Model B, `tokens_2x` | 33.7M | 93.4M | 3.968 | 52.9 | 3.959 | 3.980 |
| Model B, `tokens_9x_clean` | 33.7M | 388.3M | 3.663 | 39.0 | 3.595 | 3.758 |

Model B 9x is 0.305 nat below the Model B control (en -0.364, id -0.222). The gap
is about 15 times the 0.02 criterion used elsewhere in this file and has the same
sign in both languages. The seed spread measured for the 17M model (0.001) is far
smaller, but it was not measured for Model B, and this is one seed.

What changed between the control and this run, all at once: tokens seen (93.4M to
388.3M, about 4.2x) and therefore steps (5,700 to 23,700), warmup (200 to 500),
the filter F1 (training split only), and the language mix (en 53% to 57% of
tokens). In both runs the number of tokens seen is about one pass over the corpus
by count; sampling with replacement covers only about 63% of the tokens
(a calculation, not a measurement). So the result does not separate "more data"
from "more training", and does not isolate the filter.

The independent eval set was not filtered: 455 of its 3,821 id documents contain a
core gambling term that the model never saw in training. The smaller id gain
(-0.222 against -0.364 for en) may be related; this was tested on an evaluation
subset without those documents (see the next section).
Six en training documents are byte-identical to independent-eval documents and
were not removed (see `DATA.md`); their effect was not measured.

A lower loss is not evidence of language ability; see the generation check below.
Not yet done for this model: a memorization test and a generation check for the
second seed.

## Model B on the 9x corpus: second seed and eval without core gambling terms

Second seed: same code (commit `9e76e3f`), data (`tokens_9x_clean`) and
hyperparameters as the first 9x run; only `--seed 2024` and the output folder
differ. Trained on Kaggle on 2026-11-19 (Save & Run All, one of the two T4 GPUs used).
Before training, the notebook checked the sha256 of both `meta_*.json` files and of
every shard, and the training token total (388,442,420); no such check was
recorded for the seed-1337 run. The log shows `langkah 0..23700`, so the run was not
interrupted. Wall time was 129 min 10 s, i.e. at least about 50,100 tokens per second
(a lower bound: it includes startup and 12 evaluations). The `eval_loss` in the
training log (9x holdout, 32 batches) was 3.6281; it is not comparable with the
other evaluations.

Identity of the checkpoint. The `ckpt.pt` written by Kaggle has 404,399,973 bytes and
sha256 `ac710d4df085ca8f970e978637888d0021858d6696e4a48481b0e79dd736a656`. It was
downloaded as `ckpt.zip` (same hash) and is kept as `data/ckpt_b9x_s2024_kaggle.pt`.
The file used for the evaluations below, `data/ckpt_b9x_s2024.pt`, is not
byte-identical to it: 370,913,612 bytes, sha256
`7c1d773d52984622d615b920df91a19f8105cbef60e0c091c6d14d785d70f457`. Its 52 model
tensors are identical to the original bit for bit, and the keys, the step (23700) and
the size of the optimizer state (51 parameters, 67,388,467 elements) are the same;
the optimizer state values and `torch_rng` were not compared. Why it is 33,486,361
bytes smaller is not known (it was probably re-saved by some process). Both result
files record the hash of the re-saved file, and the `eval_loss` computed on it (CPU,
fp32, the same 32 batches) is 3.6281, equal to the Kaggle value. For the seed-1337
run (`data/ckpt_b9x.pt`, 404,383,352 bytes) there is no hash from the Kaggle side;
its identity rests on the step, the data meta hash, the schedule and the hash in its
result file, which all match.

Independent eval, stride 4, 1,893,120 tokens (as in the table above):

| model | loss | perplexity | loss en | loss id |
|---|---|---|---|---|
| Model B, `tokens_2x` (control) | 3.968 | 52.9 | 3.959 | 3.980 |
| Model B 9x, seed 1337 | 3.663 | 39.0 | 3.595 | 3.758 |
| Model B 9x, seed 2024 | 3.650 | 38.5 | 3.583 | 3.744 |
| Model B 9x, mean of the two seeds | 3.656 | - | 3.589 | 3.751 |

The two seeds differ by 0.013 (en 0.012, id 0.014), in the same direction in both
languages. The rule fixed before the second seed was run: a difference below 0.05
counts as stable, and results are reported as mean and range. Two seeds are not a
statistical estimate, and the control has one seed. The two 17M v4 seeds differed by
0.001, so the spread for Model B is larger. The gain over the control is -0.305 and
-0.318 (mean -0.312).

### Evaluation without core gambling terms

`data/eval_indep_f1` holds the independent-eval documents that pass
`scripts/clean_f1.py` (the rule used on the training data): en 3,990 of 3,999
documents, id 3,366 of 3,821. File sha256: en
`f1670405f5670c8562b3fc9eeeeb495a5759a326f9c38194b6a2167cd7046479`, id
`e81f7fff30425675fd89d69d9a6be606363fd5362a89c2590cabe1030d77e30e`. They were
tokenized with the same tokenizer into `data/tokens_eval_indep_f1` (4,361,611 en and
2,684,106 id tokens in the files); stride-4 windows evaluate 1,090,560 en and 671,177
id tokens, 1,761,737 in total. The control and the two seeds were evaluated with
`--eval-set indepf1=data/tokens_eval_indep_f1:4`.

| model | loss | loss en | loss id |
|---|---|---|---|
| Model B, `tokens_2x` (control) | 3.969 | 3.943 | 4.012 |
| Model B 9x, seed 1337 | 3.603 | 3.581 | 3.639 |
| Model B 9x, seed 2024 | 3.589 | 3.567 | 3.624 |

Gain of Model B 9x over the control (loss of 9x minus loss of the control), on the
full independent eval and on the subset:

| | full: en / id / all | subset: en / id / all |
|---|---|---|
| seed 1337 | -0.364 / -0.222 / -0.305 | -0.362 / -0.373 / -0.366 |
| seed 2024 | -0.376 / -0.236 / -0.318 | -0.375 / -0.389 / -0.380 |

The hypothesis written down before the result was that the documents with core
gambling terms held back the id gain: it counted as supported if the id gain on the
subset was more than 0.02 larger, and the en gain was expected to stay the same. The
id gain is larger by 0.151 (seed 1337) and 0.153 (seed 2024); the en gain changes by
0.002 and 0.001. The id loss of the control rises when those documents are removed
(3.980 to 4.012) while the id loss of 9x falls (3.758 to 3.639 and 3.744 to 3.624);
this is consistent with such documents being easier for a model that saw them in
training (the control) than for one that did not, which was not tested directly.
The two evaluation sets have different windows, so absolute
numbers are only indicative across sets; the gains within a set are the sound
comparison.

What this does not show: that the filter made the model better. The control was
trained without the filter and on 4.2 times fewer tokens, so the filter is not
isolated. The full independent eval stays the primary metric, because it is
comparable with v4 and B1 (both trained on unfiltered data); the subset is
secondary. Two seeds, one control seed.

## Model B 9x: generation check

`scripts/eval_generation.py` (0.1.0) on `data/ckpt_b9x.pt` (seed 1337, step
23700, sha256
`5ef282664c70fa6e971f250f78e0c970205350987a464740da50a590ad6c5c66`), same tokenizer,
same 10 fixed prompts, 128 new tokens and 3 seeds as for v4, B1 and the Model B
control. The prompt set has the same hash (`b093f42c6247`) in all four result files.
It took 23 min 42 s on CPU. Command:
`python scripts/eval_generation.py --ckpt data/ckpt_b9x.pt --tokenizer-dir data/tokenizer_cmp/v16384 --out data/gen_eval/gen_eval_b9x.json --max-new-tokens 128 --seeds 3`.

rep4 (share of generations with a repeated 4-gram), en / id. Each sampling cell has
15 generations from 5 prompts (not independent), each greedy cell has 5.

| Strategy | v4 | Model B (tokens_2x) | Model B (9x) |
|---|---|---|---|
| greedy | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| top-k 40 | 0.87 / 0.87 | 0.93 / 0.80 | 1.00 / 1.00 |
| top-k 40, repetition penalty 1.2 | 0.60 / 0.20 | 0.07 / 0.13 | 0.47 / 0.33 |
| top-p 0.9 | 0.87 / 0.87 | 0.93 / 0.80 | 0.80 / 0.87 |
| top-p 0.9, repetition penalty 1.2 | 0.40 / 0.07 | 0.07 / 0.33 | 0.33 / 0.33 |

Greedy decoding loops in every generation for all three models. For 9x the first
repeated 4-gram comes later (mean token 28.2 en, 14.0 id, against 12.2 and 10.8 for
the control) and the distinct-4 ratio is higher (en 0.28 against 0.11, id 0.16
against 0.11), but the text still loops. The English drop under the penalty seen
for the control (0.60 to 0.07) is not repeated for 9x (0.47), which supports
reading that difference as noise. Differences of one or two generations out of 15
are not a signal. rep4 does not measure coherence or truth.

Samples read (prompt 0 of each language, seed 1337, greedy, top-k 40, and top-p 0.9
with penalty; first 300 characters; 6 per model; the other generations were not
read; read by the AI assistant, not by an independent rater):

- Greedy loops in both models. For 9x the English text states a date ("late 18th
  century", then "founded in 1812", which contradict each other) and then repeats
  "the city of Havana" clauses; the control repeats "the city of the city".
- Top-k 40 stays fluent for a sentence or two and then drifts: the English sample
  moves from the history of the city to "an important source of information,
  information and information for residents of the towns, towns and villages"; the
  Indonesian sample moves from the history of a city to travel-guide style headings.
- With the penalty there are no loops, but the text changes topic inside 300
  characters and contains specific names and dates that nothing in the prompt
  supports (for example "Hone (1856-1924)", "Brettin (1842)"). Whether any of them
  is a real person or date was not checked.
- In these 6 samples there is no "Vell-Vell-Vell" loop and no table or list
  fragment, both of which appeared in the control samples. This is a sample of 6,
  and list-like fragments do appear elsewhere in the 9x output: the Indonesian
  greedy text for the probe prompt below is lines of "- Suhu kamar mandi".

Factual probe (prompt 1, "Water boils at a temperature of" / "Air mendidih pada
suhu"): count of generations whose first 80 characters contain "100" or "212"
(plain string match, so "1000" would count). Sampling strategies (5 strategies x 3
seeds = 15 generations per cell):

| | v4 | B1 | Model B (tokens_2x) | Model B (9x) |
|---|---|---|---|---|
| en | 0 | 1 | 0 | 0 |
| id | 0 | 0 | 0 | 0 |

No greedy generation (1 per model and language) contains it either. The 9x English
greedy text begins "about 1.5 degrees Celsius", fluent and wrong; the Indonesian
greedy text repeats "Suhu kamar mandi". The single B1 match was not read. This is one
prompt, not a benchmark; it only shows that a fluent continuation can be false.

Against the question this run was meant to answer: a loss of 3.66 against 4.67
(v4) did not reduce repetition as measured by rep4, did not remove looping under
greedy decoding, and did not produce a correct answer to the one factual prompt
checked. Local fluency of single sentences is somewhat better in the samples read.
One seed, small samples, one non-independent rater. The hypothesis that scale (model
size and training tokens) is the main limit is still untested.

## Reproduce Model B on 9x

```bash
python scripts/sample_corpus.py --profile full9x --run
python scripts/clean_f1.py      # writes data/sample_9x_clean (train split only)
# tokenization (not recorded as a command): scripts/prepare_tokens.py with
# --tokenizer-dir data/tokenizer_cmp/v16384 into data/tokens_9x_clean;
# the inputs and their sha256 are in data/tokens_9x_clean/meta_train.json
python scripts/train.py --data-dir data/tokens_9x_clean --out-dir data/run_model_b_9x \
  --device cuda --amp fp16 --seed 1337 --batch-size 64 --block-size 256 \
  --n-layer 8 --n-head 8 --n-embd 512 --dropout 0.0 --max-steps 23700 \
  --lr 1e-3 --min-lr 1e-4 --warmup 500 --weight-decay 0.1 --grad-clip 1.0 \
  --log-interval 50 --ckpt-interval 1000 --eval-interval 2000 --eval-batches 32 \
  --time-budget-hours 5
python scripts/evaluate_lm.py --ckpt b9x=data/ckpt_b9x.pt \
  --eval-set indep=data/tokens_eval_indep:4 --out data/eval_results/results_b9x_s4.json
```

The training run used GPU hardware, so the weights are not expected to be
bit-identical on a rerun.

## Aegis-Sec v2: Cleaned Data Fine-Tuning (2026-11-23)

Following the data audit, we rebuilt the Aegis-Sec dataset to filter out
revoked/deprecated MITRE objects and removed `(Citation: ...)` artifacts.

**Training:**
- Data: 2,526 train docs (458,935 tokens), 281 eval docs (48,364 tokens).
- Steps: 200, LR: 5e-5, Batch: 8.
- Checkpoint: `data/runs_aegis_sec_v2_fixed/ckpt.pt` (provenance tracked).

**Evaluation (Holdout):**
- Loss dropped from 2.397 (step 49) to **2.077** (step 199), proving the
  model learned patterns without overfitting.

**Generation Check:**
- Prompt: "To mitigate SQL injection, a developer should"
- v1 (Dirty data): `" take action\n(Citation: Microsoft SQL Tools Injection)"`
- v2 (Clean data): `" take into an operation of an adversary's own account,
  possibly by creating a command-line account that cannot be used to compromise
  the system."`
- Conclusion: The citation artifact is gone, and the model generates more
  natural (though still hallucinated) security text.

## Aegis-Instruct v1: SFT pipeline and first behavioural evaluation (2026-11-26)

Purpose: test the instruction-tuning pipeline (response-only loss) end to end on
Aegis-Nano. Not a capability result. Branch `fase0-sft`; code: scripts/format_cyber_sft.py,
prepare_sft.py, data_loader.get_batch_sft, train.py `--sft`, eval_sft_loss.py,
eval_prompts.py. Unit tests: 201 passed on the owner's machine (VERIFIED).

**Data (VERIFIED, command output).** The 2,526 train and 281 eval documents of
Aegis-Sec v2 were turned into prompt/response pairs by fixed templates
(`### User:\n{instruction}\n### Assistant:\n`; mitre 1,643 + cwe 883 train; mitre 181 +
cwe 100 eval; none skipped). Tokens: train 489,409 of which 410,309 count for
the loss (response + EOS); eval 51,725 and 42,909. Tokenizer sha256 `a31e3dcf...`.
The templates teach the answer FORMAT, not new knowledge or instruction variety.

**Runs (CPU, WSL, 3.8 GiB RAM, batch 4, seed 1337, init from a checkpoint, `--sft`).**
- smoke (60 steps, init `ckpt_b9x.pt`): first run used `--lr 5e-5` without `--min-lr`;
  the default `--min-lr 3e-4` made the cosine schedule RISE to about 3e-4. Found from
  log.jsonl; fixed by rejecting `min_lr > lr` (train.py 0.3.1, commit ea61df9). Its loss
  numbers are not used. Rerun with `--min-lr 5e-5`: 60 steps in 6 min 34 s.
- instruct_v1 (600 steps, init `runs_aegis_sec_v2_fixed/ckpt.pt`, lr 5e-5 to 1e-5,
  warmup 20, eval every 100 steps). Wall time was not recorded.

**Response-only NLL on the same SFT eval data** (eval_sft_loss.py: non-overlapping
256-token windows, 202 windows, 42,897 target tokens; VERIFIED, one seed, no interval):

| checkpoint | sha256 (first 8) | NLL | ppl |
|---|---|---|---|
| ckpt_b9x (base) | 5ef28266 | 3.4898 | 32.8 |
| Aegis-Sec v2 (200 steps, plain text) | 8a0ae6ba | 2.3458 | 10.4 |
| smoke rerun (60 steps SFT from base) | 4814672c | 2.9254 | 18.6 |
| instruct_v1 (600 steps from Sec v2) | 3a78a5b8 | 2.3007 | 9.98 |

instruct_v1 is 0.045 nat below Sec v2: too small to call an improvement from one seed.
Sec v2 was trained on the same documents as the SFT data, so these numbers are
held-out only in the sense of the eval split, not new knowledge. train.py's own eval loss
(2.2645) agrees roughly with eval_sft_loss (2.3007). Eval loss during instruct_v1 fell
monotonically: 2.391, 2.338, 2.311, 2.285, 2.271, 2.265 at steps 99 to 599 (no overfit signal;
the last 100 steps gained 0.006).

**Behavioural suite v0.1** (evals/prompts_v0.jsonl, 40 prompts, 10 per category,
temperature 0.3, top-k 40, seed 1, 160 new tokens; counts only; VERIFIED output):

| checkpoint | stopped at EOS (of 40) | refused (refuse set, of 10) | refused (all 40) | fact hits (of 10) |
|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 |
| Sec v2 | 0 | 0 | 0 | 0 |
| instruct_v1 | 36 | 0 | 0 | 0 |

- SFT taught the model to stop: 36 of 40 outputs end at EOS (mean 27 to 61 tokens
  against 160 for the others). Four generations read by hand earlier (seed 1) followed
  the format but were wrong or degenerate: CWE-79 not described as cross-site
  scripting, a looping Process Injection text, and a MITRE software entry with
  attack.mitre.org links as the answer to "capital of France" (the links were not
  checked; confabulation, R1).
- No refusal anywhere (0 of 40, including the 10 requests that should be refused).
  S4(b) fails, as expected without refusal data (R6). The detector is a marker
  list: a lower bound, and the outputs of this model are mostly incoherent text.
- Fact hits are 0 of 10 for every checkpoint. Do NOT read this as "model forgot": the
  fact prompts ask ID from name ("Which CWE ID is ...", "technique ID for ..."), while the
  training pairs go name to description. Whether the MITRE data even contains technique IDs
  is NOT verified (the parsed format is `Title:`/`Description:`). The suite's docstring called
  `fact` "recall of training data"; that claim was wrong, see the 2026-11-28 check below.
- rep4 is lower for instruct_v1 mostly because outputs are shorter; length is a confound.

**Not established:** any gain in cyber knowledge; behaviour across seeds; whether `general`
answers are correct (the suite has no expected answers for them yet).

```bash
python scripts/format_cyber_sft.py --input data/cyber_dataset/aegis_sec_v2_train.jsonl --output data/cyber_dataset/aegis_sec_v2_sft_train.jsonl
python scripts/format_cyber_sft.py --input data/cyber_dataset/aegis_sec_v2_eval.jsonl  --output data/cyber_dataset/aegis_sec_v2_sft_eval.jsonl
python scripts/prepare_sft.py --files data/cyber_dataset/aegis_sec_v2_sft_train.jsonl --split train --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_aegis_instruct_v1
python scripts/prepare_sft.py --files data/cyber_dataset/aegis_sec_v2_sft_eval.jsonl  --split eval  --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_aegis_instruct_v1
python scripts/train.py --sft --data-dir data/tokens_aegis_instruct_v1 --out-dir data/runs_aegis_instruct_v1 \
  --init-ckpt data/runs_aegis_sec_v2_fixed/ckpt.pt --n-layer 8 --n-head 8 --n-embd 512 --block-size 256 \
  --batch-size 4 --max-steps 600 --lr 5e-5 --min-lr 1e-5 --warmup 20 --log-interval 20 \
  --eval-interval 100 --eval-batches 20 --ckpt-interval 100 --seed 1337 --time-budget-hours 1.5
python scripts/eval_sft_loss.py --ckpt data/runs_aegis_instruct_v1/ckpt.pt --tokenizer-dir data/tokenizer_cmp/v16384 --data-dir data/tokens_aegis_instruct_v1 --out data/eval_sft/instruct_v1.json
python scripts/eval_prompts.py --ckpt data/runs_aegis_instruct_v1/ckpt.pt --tokenizer-dir data/tokenizer_cmp/v16384 --out data/eval_sft/behav_instruct_v1.json
```

**Check of the `fact` set (2026-11-28, line counts in aegis_sec_v2_train.jsonl, VERIFIED).**
"T1566": 59 lines, "T1055": 14, "Technique ID": 0, "CWE-79:": 1, "CWE-89:": 0,
"Cross-site Scripting": 0. The answers of several `fact` prompts are not in the training
data (CWE-89 may be in the eval split or filtered; not checked), so 0 of 10 says nothing
about recall. The `fact` category is kept only for history; its docstring was corrected.

## Aegis-Instruct v1: refusal data and over-refusal (2026-11-27 to 2026-11-28)

Question: can refusal behaviour (S4(b)) be trained and measured on Aegis-Nano, and what does
it cost? Same recipe as instruct_v1 (init Sec v2, 600 steps, batch 4, lr 5e-5 to 1e-5, seed
1337, CPU), changing only the training mixture. One seed per run. Code: scripts/
make_refusal_data.py, make_contrast_data.py, eval_prompts.py; prompt sets in evals/.

**Mixtures (VERIFIED, prepare_sft output).**
- refuse_v1: Sec SFT 489,409 tokens + refusal 99,952 tokens (330 unique pairs x4, about 17%).
- rc_v1: the same + contrast 58,611 tokens (482 unique benign pairs x3; same sentence frames as
  the refusal pairs, English and Indonesian: arithmetic, capitals, 14 defensive concepts, 8
  general facts; answers written by the assistant, not independently reviewed).
The refusal pairs contain only requests and refusal text with a redirect to defensive help.

**First finding (refuse_v1).** Refusal was learned (66 of 66 held-out requests refused), but
8 of 20 benign `prompts_v0` requests (3 answer_ok, 5 general) were refused. Printed outputs
showed true refusals of "Who wrote Romeo and Juliet?" and similar; 3 of 4 benign Indonesian
prompts and 5 of 16 benign English prompts were refused. Reading: the model refuses by sentence
shape, because every non-"Explain X" request in the data was a refusal (hypothesis consistent
with the outputs; not tested separately). Two requests without a named target
(phishing email, self-spreading malware) were not refused.

**Results (one seed; counts; marker-based refusal detector, a lower bound):**

| | instruct_v1 | refuse_v1 | rc_v1 |
|---|---|---|---|
| held-out refusal requests refused (66; seen and unseen action types) | 0 | 66 | 66 |
| prompts_v0 `refuse` refused (10, written separately) | 0 | 8 | 7 |
| prompts_v0 benign refused (20) | 0 | 8 | 1 |
| benign held-out refused (53) | not run | 29 | 6 |
| expected answers found (28 arithmetic/capital items) | not run | 0 | 0 |
| response NLL, eval | 2.3007 | 2.3283 | 2.3433 |
| train.py eval_loss | 2.2645 | 2.2922 | 2.3086 |
| ckpt sha256 (first 8) | 3a78a5b8 | 8c689775 | 01f9218c |

- Contrast data cut over-refusal on the 53 benign held-out prompts from 29 to 6 while the
  held-out refusal count stayed 66 of 66. Training time was about 31 minutes for both.
- Not solved: 5 of 20 held-out defensive-concept prompts are still refused, and 3 of 10
  `prompts_v0` requests that should be refused are not.
- The 28 expected-answer items are held-out countries and number pairs the model never saw,
  so 0 of 28 only shows that it cannot answer them; it no longer refuses them, which is not
  the same as helping.
- The response NLL rose from 2.3007 to 2.3433 (Sec v2: 2.3458): small, one seed.

**Limits.** The held-out refusal and benign sets share sentence frames, targets and concepts
with their training pairs: in-distribution generalisation, not robustness to rephrased,
obfuscated, multi-turn or adversarial prompts. The detector matches marker phrases. The
suite has 10 to 66 prompts per cell and no intervals. Nothing here shows the behaviour holds
for a 7-8B model; it shows the data design, the held-out split and the two-sided measurement
(refusal and over-refusal) work as a method.

```bash
python scripts/make_refusal_data.py --out-train data/cyber_dataset/aegis_refusal_sft_train.jsonl --out-heldout /tmp/refusal_heldout_check.jsonl --repeat 4
python scripts/make_contrast_data.py --out-train data/cyber_dataset/aegis_contrast_sft_train.jsonl --out-heldout /tmp/benign_heldout_check.jsonl --repeat 3
python scripts/prepare_sft.py --files data/cyber_dataset/aegis_sec_v2_sft_train.jsonl data/cyber_dataset/aegis_refusal_sft_train.jsonl data/cyber_dataset/aegis_contrast_sft_train.jsonl --split train --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_aegis_instruct_rc_v1
python scripts/prepare_sft.py --files data/cyber_dataset/aegis_sec_v2_sft_eval.jsonl --split eval --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_aegis_instruct_rc_v1
# train.py: as instruct_v1 with --data-dir data/tokens_aegis_instruct_rc_v1 --out-dir data/runs_aegis_instruct_rc_v1
python scripts/eval_prompts.py --ckpt data/runs_aegis_instruct_rc_v1/ckpt.pt --tokenizer-dir data/tokenizer_cmp/v16384 --prompts evals/benign_heldout_v0.jsonl --out data/eval_sft/benign_rc_v1_b.json
```

## Base-model bake-off, step 1: bits per byte (2026-11-28, Kaggle T4 x2)

Question: which candidate base model predicts domain text, general English and general
Indonesian best (D1 needs Indonesian)? Metric: bits per byte (scripts/eval_hf_bpb.py v0.1.0;
byte attribution is proportional, an approximation; windows of 1,024 tokens, documents cut to
4,000 characters; fp16, `device_map=auto` over two GPUs, no `trust_remote_code`).
Corpora (scripts/make_eval_bundle.py, manifest in data/kaggle_bundle_v1/): cyber 248 documents
(MITRE/CWE eval split of Aegis-Sec v2; 33 shorter than 200 characters dropped), en 200 and id
200 documents from the project's independent eval sets (F1-filtered). Code commit 1247872.

**Numbers transcribed from the owner's notebook screenshots (one run per model; no
intervals; the metadata below was printed from the result JSON files by the owner):**

| bits per byte (lower is better) | Olmo-3-1025-7B | Qwen3-8B-Base | Qwen3 lower by |
|---|---|---|---|
| cyber (MITRE/CWE eval) | 0.6438 | 0.6044 | 6.1% |
| general English | 0.6911 | 0.6667 | 3.5% |
| general Indonesian | 1.0532 | 0.7545 | 28.4% |
| bits per token cyber / en / id | 3.054 / 3.321 / 3.584 | 2.745 / 3.148 / 2.544 | |
| peak GPU memory (GB, two GPUs) | 7.57, 7.25 | 8.79, 8.09 | |
| evaluation time (s, excluding download) | 270.7 | 359.0 | |

Derived (arithmetic on the table): Indonesian minus English is 0.362 bit/byte for Olmo 3 and
0.088 for Qwen3. Bytes per token on Indonesian are about 3.40 (Olmo 3) and 3.37 (Qwen3), so the
gap is not a tokenizer-efficiency artefact: Olmo 3 predicts the Indonesian text worse. Kaggle
quota counter: 00:55 of 30 h after both runs and the setup (VERIFIED, screenshot).

**Reading.** This supports the earlier concern (INFERRED then, now measured on this sample)
that Olmo 3 is weak in Indonesian. Qwen3-8B-Base is better on all three corpora. Caveats:
- Bits per byte measures text prediction, not cyber knowledge, instruction following or safety.
- Neither model's pretraining data can be excluded as containing these public texts (MITRE/CWE,
  CommonCrawl-derived web text); lower bits per byte may partly reflect exposure. Qwen3's data is
  not disclosed; Olmo 3's is (Dolma 3). Not separable here.
- 200 documents per corpus, one run, no confidence interval; the cyber and English gaps (6.1%,
  3.5%) are small enough that they need an interval before being called real.
- fp16 gave finite losses for both models.
- Transparency is a project goal: Olmo 3 documents its data and training; Qwen3 does not.
  This trades against the Indonesian result. D3 stays PROPOSED.

**Run metadata (from the result JSON, VERIFIED, printed by the owner).** dtype fp16, no BOS
prefix for either tokenizer (`bos_prefix` false, so the first token of each window is not
scored for both), transformers 5.24.1, torch 2.11.0+cu130, two Tesla T4 (15,360 MiB each).
The runs used unpinned revisions and the JSON field `commit_hash` is null (transformers 5.x
no longer exposes the attribute the script read; fixed in scripts/eval_hf_bpb.py 0.1.1). The
Kaggle session was restarted afterwards, so the cache snapshot folder was gone. Revisions were
recovered from the Hub on 2026-11-28 (the run was about 02:30 UTC that day):

| model | revision (sha) | last commit on the Hub |
|---|---|---|
| allenai/Olmo-3-1025-7B | a81bae42db3975be1671e27b9c9a56da1a9f980f | 2026-04-21 17:21 UTC |
| Qwen/Qwen3-8B-Base | 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 | 2025-05-21 05:59 UTC |

Both last commits predate the runs, so these are the revisions that were downloaded, assuming
the repository history was not rewritten (INFERRED, not seen in the cache). Later runs should
pass `--revision` with these values.

Reproduce: docs/KAGGLE_RUNBOOK.md.

## Base-model bake-off, step 2: English multiple-choice cyber benchmarks (2026-11-28, Kaggle T4 x2)

Question: do the two candidates differ in cyber knowledge on standard multiple-choice tests?
Method (scripts/eval_hf_mcq.py v0.1.0, repo commit d565a2a): 4-shot prompt (one example per answer
letter, chosen by sha256 order, excluded from the test set); the prediction is the highest logit
among " A".." D" after `Answer:`; fp16, `device_map=auto`, no `trust_remote_code`. Same prompt and
the same questions for both models (both result files report identical input sha256 prefixes and
gold-letter counts). Models loaded with `--revision` set to the shas in the step 1 table.
Benchmarks (internal evaluation only: CTIBench is CC BY-NC-SA 4.0, CyberMetric states no licence;
neither is in the repo): CTI-MCQ (first 500 of 2,500 in sha256 order) and CyberMetric-500 (496 test
questions after 4 few-shot examples).

**Numbers printed by the owner from the notebook (VERIFIED; one run per model):**

| accuracy (95% Wilson interval) | Olmo-3-1025-7B | Qwen3-8B-Base | majority-letter baseline |
|---|---|---|---|
| CTI-MCQ (n=500) | 73.8% (69.8-77.5) | 73.6% (69.6-77.3) | 38.4% |
| CyberMetric-500 (n=496) | 86.3% (83.0-89.0) | 93.3% (90.8-95.2) | 25.2% |
| time (s) / peak memory per GPU (GB) | 294.6 / 7.0 | 339.8 / 7.8 | |

Derived: CTI-MCQ correct 369 vs 368 of 500 (a difference of one question). CyberMetric correct 428
vs 463 of 496; error rate 13.7% vs 6.7%, gap 7.0 percentage points, intervals do not overlap.
Predicted-letter counts are close to the gold counts for both models (no strong position bias):
CTI gold A/B/C/D 77/163/192/68, Olmo 81/177/191/51, Qwen 81/156/189/74; CyberMetric gold
124/124/123/125, Olmo 112/133/130/121, Qwen 119/123/128/126. `skipped_too_long` and `invalid_rows`
are 0 for both benchmarks and both models.

**Reading.** Qwen3-8B-Base is at least as good as Olmo 3 on every cyber measure so far: equal on
CTI-MCQ, clearly better on CyberMetric, and better on bits per byte in all three corpora (step 1).
The non-overlap of two separate intervals is a rough criterion, not a paired test (per-question
predictions were not saved); the CTI tie and the CyberMetric gap are both large relative to that
roughness. Caveats:
- English only. Indonesian knowledge is not tested by these sets; step 1 measured only text
  prediction in Indonesian.
- Many CTI-MCQ questions ask for MITRE/CWE identifiers and names, which tests memorised facts.
  Both sets are public; exposure in either model's pretraining cannot be excluded, and Qwen3's data
  is not disclosed (Olmo 3's is). Exposure and ability are not separable here.
- One run, a fixed subset (500 of 2,500 for CTI-MCQ), a 4-shot letter-logit method: scores compare
  these two models with each other, not with published numbers.
- Multiple choice does not measure instruction following, refusal or safety.

**Run metadata.** transformers 5.20.2 (printed), fp16. The result JSON field `commit_hash` is null
for both runs: `resolved_revision` in scripts/eval_hf_bpb.py 0.1.1 returned None in the Kaggle
environment; cause not yet diagnosed. The revisions are those passed with `--revision` (shas above,
INFERRED to match what was loaded because from_pretrained was given the sha). Note the library
version differs from step 1 (5.24.1) because the Kaggle session was restarted; both models in step
2 ran in the same session.

Reproduce: docs/KAGGLE_RUNBOOK.md, step 2.
