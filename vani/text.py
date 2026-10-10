"""Text normalisation and error rates, the same for every system.

Normalisation: Unicode NFC, drop punctuation (including the danda) and format characters
(zero-width joiners), lower-case, collapse spaces. Word error rate (WER) and character error
rate (CER) are corpus-level: total edits over total reference words or characters. CER matters
for Indian languages, where one long word can carry what English says in several.
"""
import re
import unicodedata

_SPACE = re.compile(r"\s+")


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text
                   if unicodedata.category(c) != "Cf")
    return _SPACE.sub(" ", text.lower()).strip()


def edits(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_rates(refs, hyps):
    """{"wer", "cer"} over parallel lists of raw texts."""
    we = wn = ce = cn = 0
    for r, h in zip(refs, hyps):
        r, h = normalize(r), normalize(h)
        we += edits(r.split(), h.split())
        wn += len(r.split())
        ce += edits(r.replace(" ", ""), h.replace(" ", ""))
        cn += len(r.replace(" ", ""))
    return {"wer": we / max(wn, 1), "cer": ce / max(cn, 1)}


def script_share(text, lo, hi):
    """Share of a text's letters (category L* or M*) inside the code-point block [lo, hi]."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters) if letters else 0.0
