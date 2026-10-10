"""FLORES-200 (Meta, CC BY-SA 4.0): the same 1,012 sentences (devtest) professionally translated
into 200 languages, so token counts compare like for like. FLEURS records people reading
FLORES sentences, so its transcripts are FLORES text."""
import os
import tarfile
import urllib.request

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
CACHE = os.environ.get("VANI_CACHE", os.path.expanduser("~/.cache/vani"))


def root():
    d = os.path.join(CACHE, "flores200_dataset")
    if not os.path.isdir(d):
        os.makedirs(CACHE, exist_ok=True)
        tgz = os.path.join(CACHE, "flores200_dataset.tar.gz")
        if not os.path.exists(tgz):
            urllib.request.urlretrieve(URL, tgz + ".part")
            os.replace(tgz + ".part", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(CACHE, filter="data")
    return d


def sentences(code, split="devtest"):
    with open(os.path.join(root(), split, f"{code}.{split}"), encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
