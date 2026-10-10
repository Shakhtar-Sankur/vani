"""M0: how today's speech recognisers do on Odia (FLEURS test, 883 recordings), on one GPU.

Systems, each pinned to a revision:
  OpenAI   whisper-large-v3, whisper-large-v3-turbo, whisper-small. Whisper has no Odia language
           token, so it runs with language detection; and, as the most generous setting, told
           the audio is Bengali (the closest language it has) with its Bengali-script output
           mapped letter for letter onto Odia script (the two Unicode blocks are parallel).
  Meta     mms-1b-all (CTC, Odia adapter) and seamless-m4t-v2-large (speech-to-text, Odia).

Per system: word and character error rate (vani/text.py, the same normalisation for all),
the share of output letters in Odia script, the most common detected language, seconds of
compute per second of audio, and for decoders the tokens written and how many transcripts hit
the 448-token limit. Every transcript is kept (results/m0/<lang>/<system>.jsonl).

    python scripts/eval_m0.py [--lang or_in] [--systems a,b] [--limit N] [--stream]
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vani import fleurs, text  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SYSTEMS = {
    "whisper-large-v3": ("whisper", "openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", None),
    "whisper-large-v3-as-bengali": ("whisper", "openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", "bn"),
    "whisper-large-v3-turbo": ("whisper", "openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9", None),
    "whisper-small": ("whisper", "openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", None),
    "mms-1b-all": ("mms", "facebook/mms-1b-all", "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd", "ory"),
    "seamless-m4t-v2-large": ("seamless", "facebook/seamless-m4t-v2-large", "5f8cc790b19fc3f67a61c105133b20b34e3dcb76", "ory"),
}
SCRIPT = {"or_in": (0x0B00, 0x0B7F), "en_us": (0x0041, 0x007A), "bn_in": (0x0980, 0x09FF), "hi_in": (0x0900, 0x097F)}
MAX_NEW = 440  # Whisper: 448 positions, minus the start, language, task and timestamp tokens
LANG_TOKEN = re.compile(r"<\|([a-z]{2,3})\|>")


def bengali_to_odia(s):
    """Bengali and Odia occupy parallel Unicode blocks (U+0980 and U+0B00, same layout)."""
    return "".join(chr(ord(c) + 0x180) if 0x0980 <= ord(c) <= 0x09FF else c for c in s)


def batches(items, size):
    for k in range(0, len(items), size):
        yield items[k:k + size]


def run_whisper(repo, rev, language, clips, device, batch):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(repo, revision=rev)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=dtype).to(device).eval()
    out = {}
    for chunk in batches(clips, batch):
        audio = [a for _, a in chunk]
        long = max(len(a) for a in audio) > 30 * 16000
        if long:  # sequential long-form decoding over 30-second windows
            feats = proc(audio, sampling_rate=16000, return_tensors="pt", truncation=False,
                         padding="longest", return_attention_mask=True)
        else:
            feats = proc(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        kw = dict(task="transcribe", max_new_tokens=MAX_NEW, return_dict_in_generate=False)
        if language:
            kw["language"] = language
        with torch.inference_mode():
            ids = model.generate(feats.input_features.to(device, dtype),
                                 attention_mask=feats.attention_mask.to(device), **kw)
        for (row, _), seq in zip(chunk, ids):
            raw = proc.decode(seq, skip_special_tokens=False)
            hyp = proc.decode(seq, skip_special_tokens=True).strip()
            langs = LANG_TOKEN.findall(raw)
            body = [t for t in seq.tolist() if t < proc.tokenizer.convert_tokens_to_ids("<|endoftext|>")]
            out[row["file"]] = {"hyp": bengali_to_odia(hyp) if language == "bn" else hyp, "raw_hyp": hyp,
                                "language": langs[0] if langs else None, "tokens": len(body),
                                "hit_limit": (not long) and len(body) >= MAX_NEW - 1}
    return out


def run_mms(repo, rev, lang, clips, device, batch):
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained(repo, revision=rev, target_lang=lang)
    model = Wav2Vec2ForCTC.from_pretrained(repo, revision=rev, target_lang=lang, ignore_mismatched_sizes=True)
    model.load_adapter(lang)
    model = model.to(device).eval()
    out = {}
    for chunk in batches(clips, batch):
        inputs = proc([a for _, a in chunk], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device != "cpu"):
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
        for (row, _), hyp in zip(chunk, proc.batch_decode(logits.argmax(-1))):
            out[row["file"]] = {"hyp": hyp}
    return out


def run_seamless(repo, rev, lang, clips, device, batch):
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(repo, revision=rev, dtype=dtype).to(device).eval()
    out = {}
    for chunk in batches(clips, batch):
        audio = [a for _, a in chunk]
        try:
            inputs = proc(audio=audio, sampling_rate=16000, return_tensors="pt", padding=True)
        except TypeError:
            inputs = proc(audios=audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(**inputs, tgt_lang=lang, max_new_tokens=512)
        for (row, _), seq in zip(chunk, ids):
            hyp = proc.decode(seq, skip_special_tokens=True).strip()
            out[row["file"]] = {"hyp": hyp, "tokens": int((seq != model.config.pad_token_id).sum())}
    return out


RUN = {"whisper": run_whisper, "mms": run_mms, "seamless": run_seamless}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="or_in")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stream", action="store_true", help="stream the audio archive instead of downloading it")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "m0"))
    a = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clips = list(fleurs.clips(a.lang, "test", limit=a.limit, download=not a.stream))
    clips.sort(key=lambda c: len(c[1]))  # similar lengths batch together
    audio_seconds = sum(len(x) for _, x in clips) / 16000
    lo, hi = SCRIPT.get(a.lang, (0, 0x10FFFF))
    os.makedirs(os.path.join(a.out, a.lang), exist_ok=True)
    print(json.dumps({"lang": a.lang, "clips": len(clips), "audio_seconds": round(audio_seconds, 1),
                      "device": torch.cuda.get_device_name(0) if device != "cpu" else "cpu"}), flush=True)
    summary = []  # one <system>.json per system, so runs on several GPUs can share the folder
    for name in a.systems.split(","):
        kind, repo, rev, arg = SYSTEMS[name]
        if a.lang != "or_in" and arg in ("bn", "ory"):
            continue  # Odia-specific settings
        t0 = time.time()
        out = RUN[kind](repo, rev, arg, clips, device, a.batch)
        if device != "cpu":
            torch.cuda.synchronize()
        seconds = time.time() - t0
        refs = [r["raw"] for r, _ in clips]
        hyps = [out[r["file"]]["hyp"] for r, _ in clips]
        rates = text.error_rates(refs, hyps)
        row = {"system": name, "model": repo, "revision": rev, "wer": round(rates["wer"], 4), "cer": round(rates["cer"], 4),
               "script_share": round(sum(text.script_share(h, lo, hi) for h in hyps) / len(hyps), 3),
               "seconds": round(seconds, 1), "seconds_per_audio_second": round(seconds / audio_seconds, 4)}
        langs = collections.Counter(v.get("language") for v in out.values() if v.get("language"))
        if langs:
            row["detected_languages"] = dict(langs.most_common(5))
        if any("tokens" in v for v in out.values()):
            row["tokens_per_transcript"] = round(sum(v.get("tokens", 0) for v in out.values()) / len(out), 1)
            row["hit_448_limit"] = sum(bool(v.get("hit_limit")) for v in out.values())
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        json.dump(row, open(os.path.join(a.out, a.lang, f"{name}.json"), "w"), indent=1, ensure_ascii=False)
        with open(os.path.join(a.out, a.lang, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r, _ in clips:
                f.write(json.dumps({"file": r["file"], "ref": r["raw"], **out[r["file"]]}, ensure_ascii=False) + "\n")
        del out
        if device != "cpu":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
