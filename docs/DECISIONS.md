# Decision log (ADR)

Rule: every entry is PROPOSED until the owner writes ACCEPTED with a date.
The assistant never records an acceptance on the owner's behalf.

## D1 Target language(s) - ACCEPTED 2026-11-25 (owner's answer in session)
English + Indonesian. Impact: data, eval sets, tokenizer. Risk (INFERRED): public
cyber benchmarks found so far (CTIBench, CyberMetric) are in English, so Indonesian
evaluation sets must be written by the project; see ROADMAP, Phase 1 inputs.

## D2 Red-team scope - ACCEPTED 2026-11-25 (owner's answer in session)
S1 as written (defensive/analytical use and support for authorised testing). Working
exploits and malware stay out of scope, and outside what the assistant will help build.

## D3 Base model for Track B - PROPOSED
Owner's stated preference (2026-11-25): Olmo 3 7B; Qwen3-8B-Base as comparison. The choice is
made by the Phase 1 bake-off, not by preference. Facts (Hugging Face cards, 2026-11-28): both
Apache 2.0. Step 1 result (docs/EXPERIMENTS.md, one run each): Qwen3-8B-Base has lower bits per
byte on all three corpora, by 28.4% on Indonesian, 6.1% on cyber text, 3.5% on English. Since D1
requires Indonesian, the evidence so far favours Qwen3-8B-Base as the working base for Phase 3.
Against it: its training data is not disclosed, while Olmo 3's is (project goal: transparency).
Step 2 result (docs/EXPERIMENTS.md, English multiple-choice, 4-shot, one run each): CTI-MCQ is a
tie (73.8% vs 73.6%, 500 questions); CyberMetric-500 favours Qwen3-8B-Base (93.3% vs 86.3%,
non-overlapping 95% intervals). No measure so far favours Olmo 3 beyond noise.
Remaining before acceptance: a project-written Indonesian set (Indonesian knowledge is still not
tested directly), a reading of both licences by a qualified person (G2), and the owner's
decision on the transparency trade-off.

## D4 Compute plan - PROPOSED
Local hardware (PENDING: owner to state RAM/GPU/VRAM) plus Kaggle when
needed. Kaggle limits NOT VERIFIED (see ROADMAP). Record GPU hours per run
(R10).

## D5 Adaptation method - PROPOSED
LoRA/SFT first; continued pretraining only if a measured gap justifies it.

## D6 Agent autonomy - PROPOSED
Stages 0-3; no stage without written human approval. Policy gate lives in
code outside the model; model output returns to the model as data (R9, S6).
