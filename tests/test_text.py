import importlib.util
import os

from vani import text


def test_normalize_drops_punctuation_danda_and_zero_width_joiners():
    assert text.normalize("ଆରିଷ୍ଟୋଟଲ‌ଙ୍କ ମତ।") == "ଆରିଷ୍ଟୋଟଲଙ୍କ ମତ"
    assert text.normalize("Hello,  World!") == "hello world"


def test_normalize_keeps_vowel_signs_and_virama():
    word = "କହିଛନ୍ତି"  # vowel signs (Mc/Mn) and a virama must survive
    assert text.normalize(word) == word


def test_edits_and_error_rates():
    assert text.edits("kitten", "sitting") == 3
    r = text.error_rates(["a b c d"], ["a x c"])
    assert r["wer"] == 0.5
    assert abs(r["cer"] - 2 / 4) < 1e-9
    assert text.error_rates(["ମତ"], ["ମତ।"]) == {"wer": 0.0, "cer": 0.0}


def test_script_share():
    assert text.script_share("ଓଡ଼ିଆ abc", 0x0B00, 0x0B7F) == 5 / 8
    assert text.script_share("123 ...", 0x0B00, 0x0B7F) == 0.0


def test_bengali_maps_letter_for_letter_onto_odia():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "eval_m0.py")
    spec = importlib.util.spec_from_file_location("eval_m0", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.bengali_to_odia("কথা 12") == "କଥା 12"
