# Aegis-Core roadmap (Phase 0 draft, 2026-11-25)

Status: PROPOSED by the AI assistant for the owner's review. No phase below is
approved until the owner records it in docs/DECISIONS.md. Labels as in
docs/THREAT_MODEL.md (VERIFIED, REPORTED, INFERRED, PENDING, BLOCKER).

Goal (REPORTED, owner): an open-source, transparent, locally runnable model
for cyber security and red-team work, which may later act as a bounded agent.
Scope limit (S1): defensive/analytical use and support for authorised testing;
no working exploits or malware. No claim of "compliant", "certified" or
"production ready" (G6); only "mapped to <framework> with evidence".

## Tracks

- Track A, Aegis-Nano (33.7M): methodology lab (data, SFT, evaluation, safety
  experiments). Not expected to hold reliable domain knowledge (S3, INFERRED).
- Track B, product model: an adapted open 7-8B base chosen in Phase 1 by a
  bake-off. Candidates (INFERRED, to verify): Olmo 3 7B, Qwen3-8B-Base;
  Foundation-Sec-8B as evaluation baseline only until its licence is checked.

## Phases and exit gates

Each phase ends only when its gate is met with recorded evidence.

| # | Phase | Exit gate (all written, with hashes/counts) | Status |
|---|-------|---------------------------------------------|--------|
| 0 | Governance foundation | ROADMAP, DECISIONS, updated THREAT_MODEL merged; owner answers D1-D6 | IN PROGRESS |
| 1 | Evaluation and base choice | Fixed eval suite (6 layers) runs end to end; bake-off report; licence of chosen base read | IN PROGRESS (bake-off: bits per byte and English MCQ done; Indonesian set pending) |
| 2 | Data and licences | Each source audited per S2/DATA.md; train/eval split, near-dup and contamination check | NOT STARTED |
| 3 | Domain adaptation | LoRA/SFT first; CPT only if a gap is measured; results vs base | NOT STARTED |
| 4 | Instruction and safety tuning | S4(b) refusal set measured, failures recorded | NOT STARTED |
| 5 | Isolated capability tests | Isolated range only; written authorisation; no external targets | NOT STARTED |
| 6 | Local runtime | Documented local inference on stated hardware, measured | NOT STARTED |
| 7 | Bounded agent | S6 scope + policy gate in code outside the model + human approval + immutable log | NOT STARTED |
| 8 | Release and governance | G2-G5 passed; legal review done by a qualified person | NOT STARTED |

Time estimates are deliberately not given here; the earlier 19-34 week range
was INFERRED and is not evidence.

## Evaluation layers (Phase 1)

1. General regression. 2. Cyber knowledge benchmarks (chosen and
version-checked at that time). 3. Execution/CTF/range tasks (isolated).
4. Refusal set (S4b). 5. Privacy and contamination. 6. Confabulation.
Contamination warning (INFERRED): MITRE/CWE training text may overlap with
benchmark items built from the same sources; check before reporting scores.

## Phase 1 inputs (read 2026-11-28; Hugging Face cards and dataset pages, via fetch)

| | Olmo-3-1025-7B | Qwen3-8B-Base |
|---|---|---|
| licence (card) | Apache 2.0 | Apache 2.0 |
| size | 7B | 8.2B (6.95B non-embedding) |
| context | 65,536 | 32,768 |
| training data | Dolma 3 (stage 1: 5.93T tokens) | 36T tokens, 119 languages |
| transformers | >= 4.57.0 | >= 4.51.0 |

The Olmo 3 card warns that the models can easily be prompted to produce harmful content and
that their statements need checking; a note about research and educational use was also
reported by the fetch tool, wording not verified: needs a legal read before any release (G2).
Newer Qwen releases exist (a Qwen3.8 headline appeared in search); not evaluated here.

Public benchmarks found (S4a asks for a version check at use time):
- CTIBench: Hugging Face card says licence CC BY-NC-SA 4.0 (non-commercial, share-alike);
  configs cti-mcq, cti-rcm, cti-vsp, cti-taa (row counts not stated on the card page read);
  zero-shot CTI tasks. Use for evaluation only, not training; check the licence against the
  intended release (G2).
- CyberMetric: multiple choice A-D in sizes 80, 500, 2,000 and 10,000; built with LLMs and
  RAG and human-validated (README); no licence stated on the README page read, so treat as
  unknown until checked.
Both are English (INFERRED). Indonesian cyber evaluation needs a project-written set.
Contamination: MITRE/CWE text used for Aegis-Sec training may overlap with CTIBench items
(INFERRED); compare scores before and after domain adaptation with that in mind.

Model memory (INFERRED, to measure): 7-8B parameters at 2 bytes is about 14-17 GB of weights.
Kaggle offers T4 x2 (announcement title seen; per-GPU memory 16 GB is the NVIDIA T4
specification, not read from Kaggle) so one model can be split across two GPUs, or loaded in
4-bit on one. T4 has no native bf16, so fp16 is used; fp16 overflow risk is not tested.

## Compute

Local machine: WSL shows 3.8 GiB RAM and no GPU (VERIFIED, owner's output,
2026-11-25); the Windows host specifications are still PENDING. This is enough
for Aegis-Nano, not for 7-8B models. Kaggle free GPU is the candidate for Track B.
Owner's account page (screenshot 2026-11-26) shows "GPU 30h available of 30h";
the period (probably weekly) and session limits are NOT shown and not verified.
Kaggle documentation pages could not be read by the tooling. A Kaggle announcement
says the NVIDIA Tesla P100 is retired from 15 September 2026 (page text truncated; remaining
options not read), so plan for T4 x2. A Kaggle discussion title reports GPU availability
degrading; session length, internet access rules and phone verification are NOT verified. Inference memory for
7-8B fp16 is about 14-16 GB (INFERRED), tight on one T4 (no native bf16); to be
measured in Phase 1.

## Training rule

No training run for Track B before the Phase 1 gate. Track A SFT code exists
on branch `fase0-sft` (not merged to main; 215 tests pass on the owner's machine).
Runs recorded in docs/EXPERIMENTS.md: SFT with response-only loss, a fixed behavioural
suite, and a refusal/over-refusal experiment. These validate the method on a 33.7M model;
they do not show capability and say nothing about 7-8B models.
