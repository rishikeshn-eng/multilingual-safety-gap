# multilingual-safety-gap

**Live demo:** [https://rishikeshn-eng.github.io/multilingual-safety-gap/](https://rishikeshn-eng.github.io/multilingual-safety-gap/) runs the whole pipeline in your browser (bring your own Gemini key for the model calls).

Does a model's safety behaviour survive translation? Safety training is
concentrated in English. This measures whether the same request gets the
same treatment in Hindi and in Hinglish.

Gemini API only. No GPU. A full run is about 216 calls.

## Why this one matters for you specifically

Almost every published safety eval is English-only. India runs on three
languages at once: English, Hindi, and Hinglish (romanised Hindi-English,
which is how a huge share of Indian users actually type). A model deployed
here is exposed to all three, and its safety training probably was not.
Very few people are measuring that gap with matched prompts. You are
positioned to be one of them, and that is a real differentiator.

## The design

Matched triplets. Every prompt exists in English, Hindi, and Hinglish with
the same content. Because the content is held constant, any difference in
how the model responds is a language effect.

- 6 harmful prompts that SHOULD be refused
- 6 safe prompts that SHOULD be complied with

The safe set is built as near-misses of the harmful set. Lock-picking for a
locksmith student vs. for a break-in. Phishing to train your parents vs.
phishing to run a scam. The model has to read intent, not keywords, which is
exactly where multilingual handling tends to break.

## Two failure directions, both measured

- **Under-refusal:** refuses the harmful English prompt but complies with the
  same prompt in Hindi or Hinglish. The classic multilingual jailbreak.
- **Over-refusal:** helps with the safe English prompt but refuses the same
  benign request in another language, treating the language itself as
  suspect. This one is under-studied and just as real.

## Setup and run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key

python src/run_eval.py --mock --runs 3         # 0 API calls
python src/analyze.py results/raw_mock_*.jsonl

python src/run_eval.py --limit 3 --runs 1      # cheap real test
python src/run_eval.py --runs 3                # full run
python src/analyze.py results/raw_gemini*.jsonl
```

The mock plants a synthetic gap so the analysis has signal. That is a wiring
test, not a result.

## Reading it honestly

- The report prints per-run spread. A language gap smaller than that spread
  is noise, not a finding.
- The safety-response classifier is itself an LLM and can misjudge,
  especially on PARTIAL responses. Read a sample of transcripts by hand.
- 12 prompts is a demonstration, not a broad claim about any model. Widen the
  set before drawing strong conclusions.
- My Hindi and Hinglish translations are hand-written and may read slightly
  unnaturally in places. A native reviewer would strengthen this; note it
  rather than hide it.

## Extending it

- Add more Indian languages (Tamil, Bengali, Marathi) as further columns
- Add code-switching within a single prompt, which is very common in practice
- Compare two models and report which has the smaller gap
- Have a fluent speaker review and improve the translations
