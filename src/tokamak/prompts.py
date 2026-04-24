"""Caveman system prompts adapted for *internal reasoning* compression.

Derived from JuliusBrussee/caveman SKILL.md. The runtime caveman skill targets
all assistant output; here we target reasoning channels only and force "lite"
intensity by default — reasoning must stay legible to a downstream model that
will be trained on it.
"""

CAVEMAN_LITE_REASONING = """\
You are rewriting an assistant's internal reasoning so that the same logical
steps survive in roughly half the tokens. The reasoning will be used as
training data — no information may be lost, only filler.

Rules:
- Drop filler words: just, really, basically, actually, simply, literally,
  obviously, clearly, of course, indeed.
- Drop pleasantries and self-talk: "Sure", "Let me think", "Okay so",
  "I'll start by", "First, I want to", "Now I'm going to", "Great", "Perfect".
- Drop hedging unless it carries information: "I think", "perhaps", "maybe",
  "it seems", "I believe". Keep "if", "unless", "because", "therefore".
- Keep articles (a/an/the). Keep grammar. Lite, not full caveman.
- Short synonyms: "use" not "make use of", "fix" not "implement a solution
  for", "check" not "verify that".
- Preserve verbatim and unchanged: code blocks, file paths, shell commands,
  function names, identifiers, error messages, version strings, URLs, numbers,
  quoted user input.
- Preserve every reasoning STEP. Compress the prose around the step, never
  remove the step itself. If the original tries three approaches before
  picking one, the rewrite must still try three approaches.
- Preserve the order of steps.
- Output the rewritten reasoning ONLY. No preamble, no "Here is", no fences.

Pattern: [observation]. [inference]. [next step].
"""

CAVEMAN_FULL_REASONING = """\
You are rewriting an assistant's internal reasoning in full caveman mode for
training data. Drop ~70% of tokens. All technical substance survives.

Rules:
- Drop articles (a/an/the).
- Drop filler/pleasantries/hedging entirely.
- Sentence fragments OK.
- Short synonyms.
- Preserve verbatim: code, paths, identifiers, errors, numbers, quoted input.
- Preserve every reasoning step in original order.
- Use arrows for causality where natural: X -> Y.
- Output rewritten reasoning ONLY.

Pattern: [thing] [action] [reason]. [next step].
"""

LEVELS = {
    "lite": CAVEMAN_LITE_REASONING,
    "full": CAVEMAN_FULL_REASONING,
}


def system_prompt(level: str = "lite") -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown caveman level: {level!r}; choose {list(LEVELS)}")
    return LEVELS[level]
