# vani

**Speech recognition for Indian languages, measured against OpenAI, Meta and Google, and then
built better.** *Vani* (ବାଣୀ, वाणी) means "voice, speech" in Odia, Hindi and Sanskrit.

The world's speech recognisers are trained on about a hundred languages at once, so a language
like Odia, spoken by 37.5 million people, gets very little of them. vani measures what they
get wrong on FLEURS, a public test set of people reading the same sentences in 102 languages,
and then trains a model for Odia on free GPUs to beat them on the same test.

## M0, part 1: Whisper cannot write Odia

Before measuring accuracy, the structure of OpenAI's Whisper (large-v3) already rules most of
Odia out (`scripts/token_tax.py`, `results/m0/token_tax.json`):

- **Whisper has no Odia.** Its tokenizer has a language token for 100 languages, including 14
  Indian ones (Hindi, Bengali, Assamese, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati,
  Punjabi, Urdu, Nepali, Sindhi, Sanskrit), but not Odia, so it cannot be told what it is
  hearing.
- **Odia costs 12.7× English in Whisper's tokens.** For the same 1,012 FLORES sentences,
  Whisper's tokenizer needs 355,250 tokens in Odia against 27,991 in English. Whisper writes a
  transcript one token at a time, so the same meaning takes about 12× as many decoding steps.

  | language | tokens, ×English | | language | tokens, ×English |
  |---|---|---|---|---|
  | Odia | **12.69×** (no language token) | | Bengali | 9.13× |
  | Gujarati | 11.52× | | Telugu | 8.60× |
  | Malayalam | 11.39× | | Kannada | 8.17× |
  | Punjabi | 9.24× | | Marathi | 5.34× |
  | Assamese | 9.18× | | Hindi | 4.88× |

- **One in five Odia recordings cannot fit.** Whisper writes at most 448 tokens for each
  30-second window of audio. Odia speech in FLEURS runs at 29.8 Whisper tokens per second
  (English: 2.9), so the window is full after 15 seconds of Odia (English: 157 seconds). Of the
  875 Odia test recordings that fit in one window, **194 (22%) need more than 448 tokens**:
  no Whisper model can write those transcripts in full, however well it hears them.

## M0, part 2: accuracy, head to head (running)

`scripts/eval_m0.py` runs every system on all 883 FLEURS Odia test recordings and keeps every
transcript; `scripts/kaggle_m0.sh` runs it on Kaggle's two T4s:

| company | system | how it is run on Odia |
|---|---|---|
| OpenAI | Whisper large-v3, large-v3-turbo, small | language detection (there is no Odia token) |
| OpenAI | Whisper large-v3, told "Bengali" | the most generous setting: the closest language it has, with its Bengali-script output mapped letter for letter onto Odia script |
| Meta | MMS-1b-all | its Odia adapter |
| Meta | Seamless-M4T v2 large | Odia as the target language |

Reported per system: word and character error rate (the same normalisation for every system,
`vani/text.py`), the share of the output written in Odia script, the language Whisper thinks it
hears, compute time per second of audio, and how many transcripts hit Whisper's 448-token limit.
An English control (Whisper on 120 English recordings) checks that the harness is fair.

Google is next: its Speech-to-Text API supports Odia and needs an API key.

## Plan

| | Milestone | State |
|---|---|---|
| M0 | Measure: Whisper's token tax and window limit; OpenAI's and Meta's systems on FLEURS Odia; Google's API | token tax done; accuracy running |
| M1 | Fix Whisper's tokenizer for Odia (as in [lipi](https://github.com/Shakhtar-Sankur/lipi)): an Odia language token and Odia tokens, so a transcript fits and decodes in fewer steps | |
| M2 | Fine-tune on Kaggle's 2× T4 on open Odia speech; head-to-head on FLEURS against every M0 system | |
| M3 | More Indian languages; speed; write-up | |

## Run it

```
pip install tokenizers numpy soundfile torch "transformers==5.19.0" sentencepiece pytest
python scripts/token_tax.py                          # Whisper's token tax (CPU, about a minute)
python scripts/eval_m0.py --limit 3 --stream --systems whisper-small   # a quick check on CPU
python -m pytest -q tests
```

FLEURS (CC BY 4.0) and FLORES-200 (CC BY-SA 4.0) download into `~/.cache/vani` (`VANI_CACHE`).
Every model is pinned to a revision. Meta's MMS and Seamless are licensed for non-commercial
use (CC BY-NC 4.0); vani only evaluates them.

## License

Apache-2.0. See [LICENSE](LICENSE).
