"""Whisper's token tax: tokens Whisper's decoder must write for the same sentences in each
language (FLORES-200 devtest), and for FLEURS test transcripts, tokens per second of speech
against the decoder's 448-token window: Whisper writes at most 448 tokens for each 30 seconds
of audio, so a transcript longer than that cannot be written, however well the model hears it.

    python scripts/token_tax.py      # writes results/m0/token_tax.json
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizers import Tokenizer  # noqa: E402

from vani import fleurs, flores  # noqa: E402
from vani.flores import CACHE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m0", "token_tax.json")
# tokenizer.json at pinned revisions
WHISPER = {"whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
           "whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d"}
WINDOW = 448  # max_target_positions of every Whisper model


def tokenizer(name):
    path = os.path.join(CACHE, "whisper", f"{name}.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/openai/{name}/resolve/{WHISPER[name]}/tokenizer.json", path)
    return Tokenizer.from_file(path)


def count(tok, texts):
    return [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]


def main():
    tok = tokenizer("whisper-large-v3")
    cfg = json.load(open(os.path.join(CACHE, "whisper", "whisper-large-v3.json")))
    lang_tokens = sorted(a["content"][2:-2] for a in cfg["added_tokens"]
                         if a["content"].startswith("<|") and a["content"][2:-2].isalpha()
                         and a["content"][2:-2].islower() and len(a["content"]) <= 9)
    eng = sum(count(tok, flores.sentences("eng_Latn")))
    out = {"tokenizer": "openai/whisper-large-v3", "revision": WHISPER["whisper-large-v3"],
           "language_tokens": lang_tokens, "flores_devtest": {}, "fleurs_test": {}}
    for fl, code in fleurs.LANGS.items():
        n = sum(count(tok, flores.sentences(code)))
        out["flores_devtest"][code] = {"fleurs": fl, "whisper_language_token": fl.split("_")[0] in lang_tokens,
                                       "tokens": n, "tax": round(n / eng, 2)}
        print(f"{code}  {n:7d} tokens  {n / eng:5.2f}x English  language token: {fl.split('_')[0] in lang_tokens}", flush=True)
    for fl in ("en_us", "or_in"):
        rs = fleurs.rows(fl, "test")
        n = count(tok, [r["raw"] for r in rs])
        secs = [int(r["num_samples"]) / 16000 for r in rs]
        one = [(x, s) for x, s in zip(n, secs) if s <= 30]  # utterances Whisper decodes in one window
        tps = sum(n) / sum(secs)
        out["fleurs_test"][fl] = {
            "utterances": len(rs), "tokens_per_second": round(tps, 2), "max_tokens": max(n),
            "within_30s": len(one), "within_30s_over_window": sum(x > WINDOW for x, _ in one),
            "seconds_that_fill_window": round(WINDOW / tps, 1), "max_seconds": round(max(secs), 1)}
        print(fl, out["fleurs_test"][fl], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
