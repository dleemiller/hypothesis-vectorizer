"""Hand-written, class-tagged expert hypothesis pools + zero-shot class templates.

These are the LM-free ingredients of the study:
  * `EXPERT_POOLS[ds]`   -> list of (hypothesis_text, intended_class) — the HV-fixed-expert pool
                           and the class tags the prior-aggregation head aggregates over.
  * `ZEROSHOT_TEMPLATES[ds]` -> one entailment template per class, in class-index order —
                           the zero-shot-NLI baseline (score each, argmax entailment).

Hypotheses follow the paper's rules: short, atomic, a statement *about the text*, verifiable
from the text alone, not a bare label name, semantically diverse. `intended_class` and any
rationale are metadata for analysis and for the prior head — only the text is a feature.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TREC-6 question classification. Classes (index order matches data.py):
#   0 ABBR  1 ENTY  2 DESC  3 HUM  4 LOC  5 NUM
# ---------------------------------------------------------------------------
_TREC_EXPERT: list[tuple[str, str]] = [
    # ABBR — abbreviation / expansion
    ("The text asks what an abbreviation or acronym stands for.", "ABBR"),
    ("The text asks for the full form of an initialism.", "ABBR"),
    ("The question is about what a set of letters means.", "ABBR"),
    # ENTY — entity (thing, animal, substance, product, term, ...)
    ("The text asks for the name of a thing or object.", "ENTY"),
    ("The text asks which animal, plant, or substance is being described.", "ENTY"),
    ("The text asks for the title of a book, film, or creative work.", "ENTY"),
    ("The text asks what something is called.", "ENTY"),
    ("The text asks for the name of a color, food, or material.", "ENTY"),
    # DESC — description, definition, manner, reason
    ("The text asks for the definition of a term.", "DESC"),
    ("The text asks for an explanation of why something happens.", "DESC"),
    ("The text asks how to do something or how something works.", "DESC"),
    ("The text asks for a description of something's meaning or purpose.", "DESC"),
    # HUM — human / person / group / organization
    ("The text asks for the name of a person.", "HUM"),
    ("The text asks who did something or who is responsible.", "HUM"),
    ("The text asks about a group, team, or organization of people.", "HUM"),
    ("The text asks for the identity of an individual.", "HUM"),
    # LOC — location
    ("The text asks where something is located.", "LOC"),
    ("The text asks for the name of a place, city, or country.", "LOC"),
    ("The text asks about a geographic location.", "LOC"),
    ("The text asks which region or area something is in.", "LOC"),
    # NUM — numeric value
    ("The text asks for a number or a count.", "NUM"),
    ("The text asks how many of something there are.", "NUM"),
    ("The text asks for a date, year, or period of time.", "NUM"),
    ("The text asks for a distance, size, or measurement.", "NUM"),
    ("The text can be answered with a numeric value.", "NUM"),
]

# One entailment template per class, in class-index order, for zero-shot NLI.
_TREC_ZEROSHOT = [
    "The text asks what an abbreviation stands for.",          # ABBR
    "The text asks for the name of an entity or thing.",       # ENTY
    "The text asks for a description, definition, or reason.",  # DESC
    "The text asks about a person or group of people.",        # HUM
    "The text asks about a location or place.",                # LOC
    "The text asks for a number, quantity, or date.",          # NUM
]

# ---------------------------------------------------------------------------
# AG News topic classification. Classes: 0 World  1 Sports  2 Business  3 Sci/Tech
# ---------------------------------------------------------------------------
_AGNEWS_EXPERT: list[tuple[str, str]] = [
    ("The article is about international news, politics, or world events.", "World"),
    ("The article describes conflict, diplomacy, or a government.", "World"),
    ("The article is about a sports team, match, or athlete.", "Sports"),
    ("The article reports a game result or a sporting competition.", "Sports"),
    ("The article is about a company, market, or the economy.", "Business"),
    ("The article discusses finance, stocks, or corporate earnings.", "Business"),
    ("The article is about science, technology, or computing.", "Sci/Tech"),
    ("The article describes a scientific discovery or a new technology.", "Sci/Tech"),
]
_AGNEWS_ZEROSHOT = [
    "This article is about world news and politics.",
    "This article is about sports.",
    "This article is about business and finance.",
    "This article is about science and technology.",
]

# ---------------------------------------------------------------------------
# SST-2 sentiment. Classes: 0 negative  1 positive
# ---------------------------------------------------------------------------
_SST2_EXPERT: list[tuple[str, str]] = [
    ("The text expresses a negative opinion.", "negative"),
    ("The reviewer dislikes the film.", "negative"),
    ("The text criticizes or complains about something.", "negative"),
    ("The text conveys disappointment or frustration.", "negative"),
    ("The text expresses a positive opinion.", "positive"),
    ("The reviewer praises or recommends the film.", "positive"),
    ("The text conveys enjoyment or admiration.", "positive"),
    ("The text is enthusiastic or approving.", "positive"),
]
_SST2_ZEROSHOT = [
    "The text expresses a negative sentiment.",
    "The text expresses a positive sentiment.",
]


EXPERT_POOLS: dict[str, list[tuple[str, str]]] = {
    "trec": _TREC_EXPERT,
    "ag_news": _AGNEWS_EXPERT,
    "sst2": _SST2_EXPERT,
}

ZEROSHOT_TEMPLATES: dict[str, list[str]] = {
    "trec": _TREC_ZEROSHOT,
    "ag_news": _AGNEWS_ZEROSHOT,
    "sst2": _SST2_ZEROSHOT,
}


def expert_pool(dataset: str) -> tuple[list[str], list[str]]:
    """(hypothesis texts, intended class name per hypothesis) for a dataset."""
    pairs = EXPERT_POOLS[dataset]
    return [h for h, _ in pairs], [c for _, c in pairs]
