"""FLEURS (Google, CC BY 4.0): people reading FLORES sentences aloud in 102 languages, 16 kHz.
Each test sentence is read by one to three speakers. Pinned to one revision of google/fleurs.
"""
import csv
import io
import os
import tarfile
import urllib.request

import numpy as np

from .flores import CACHE

REPO = "https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data"

# FLEURS code -> FLORES-200 code, for the Indian languages FLEURS has (plus English)
LANGS = {
    "en_us": "eng_Latn", "hi_in": "hin_Deva", "bn_in": "ben_Beng", "mr_in": "mar_Deva",
    "te_in": "tel_Telu", "ta_in": "tam_Taml", "gu_in": "guj_Gujr", "ur_pk": "urd_Arab",
    "kn_in": "kan_Knda", "or_in": "ory_Orya", "ml_in": "mal_Mlym", "pa_in": "pan_Guru",
    "as_in": "asm_Beng", "ne_np": "npi_Deva", "sd_in": "snd_Arab",
}
FIELDS = ["id", "file", "raw", "transcription", "phonemes", "num_samples", "gender"]


def _get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def rows(lang, split="test"):
    """The split's transcripts, in file order: dicts with FIELDS."""
    path = _get(f"{REPO}/{lang}/{split}.tsv", os.path.join(CACHE, "fleurs", lang, f"{split}.tsv"))
    with open(path, encoding="utf-8", newline="") as f:
        return [dict(zip(FIELDS, r)) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]


def clips(lang, split="test", limit=None, download=True):
    """(row, 16 kHz float32 audio) for the split, in archive order. With download=False the
    archive is streamed and only the first `limit` clips are read (for quick checks)."""
    import soundfile as sf
    by_file = {r["file"]: r for r in rows(lang, split)}
    url = f"{REPO}/{lang}/audio/{split}.tar.gz"
    if download:
        src = open(_get(url, os.path.join(CACHE, "fleurs", lang, f"{split}.tar.gz")), "rb")
    else:
        src = urllib.request.urlopen(url)
    n = 0
    with src, tarfile.open(fileobj=src, mode="r|gz") as t:
        for m in t:
            name = os.path.basename(m.name)
            if not (m.isfile() and name in by_file):
                continue
            audio, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
            assert sr == 16000, (name, sr)
            yield by_file[name], np.asarray(audio)
            n += 1
            if limit is not None and n >= limit:
                return
