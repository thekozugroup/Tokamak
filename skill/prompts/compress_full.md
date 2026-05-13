You rewrite an assistant's internal reasoning in aggressive terse mode for
training data. Drop ~70% of tokens. All technical substance survives.

You are given three things:
1. PROBLEM     — user's question.
2. ANSWER      — assistant's final answer.
3. REASONING   — internal trace to rewrite.

PROBLEM and ANSWER are reference rails only. Do not rewrite them, do not cite
them. They prevent you from drifting into steps that fit a different answer.

Rules for REASONING:
- Drop articles (a/an/the).
- Drop filler / pleasantries / hedging entirely.
- Sentence fragments OK.
- Short synonyms.
- Preserve VERBATIM: code, paths, identifiers, errors, numbers, formulae,
  quoted input.
- Preserve every reasoning step in original order.
- Use arrows for causality where natural: X -> Y.
- Do not introduce new facts or alternative methods.
- Output the rewritten REASONING ONLY.

Pattern: [thing] [action] [reason]. [next step].
