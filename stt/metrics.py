"""Measure transcription quality with Word Error Rate (WER).

WER = (substitutions + deletions + insertions) / number of words in the reference

  reference:  "turn on the kitchen light"
  hypothesis: "turn on a kitchen light please"
  -> 1 substitution (the -> a) + 1 insertion (please) = 2 errors / 5 words = 40% WER

0% is perfect. Lower is better. It can go above 100% if many extra words are added.
"""

import re
from dataclasses import dataclass


@dataclass
class WERResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation so "Hello, world!" equals "hello world".

    Note: numbers are not converted, so "5" and "five" still count as different.
    """
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)  # drop quote marks, keep "don't"
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> WERResult:
    """Compare two texts word by word using edit distance (Levenshtein)."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    n, m = len(ref), len(hyp)

    # dp[i][j] = fewest edits to turn the first i reference words into the first j hypothesis words
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,          # deletion
                           dp[i][j - 1] + 1,          # insertion
                           dp[i - 1][j - 1] + cost)   # match or substitution

    # Walk back through the table to count each type of error.
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    if n == 0:
        wer = 0.0 if m == 0 else 1.0
    else:
        wer = (subs + dels + ins) / n
    return WERResult(wer, subs, dels, ins, n)
