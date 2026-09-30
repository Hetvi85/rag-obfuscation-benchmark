# Benchmarking the Retrieval–Evasion Trade-off of Obfuscated IPI in RAG

COMPSCI 764 project, Group 28. Measures how obfuscation intensity `ρ` jointly
affects retrievability `P_ret(ρ)` and guardrail evasion `ε(ρ)`, and locates the
optimum `ρ*` that maximises end-to-end attack success `ASR = P_ret · ε · P_gen`.

**Ethics:** the injected instruction only asks the model to append the harmless
marker `CANARY-7F3A`. Do not replace it with a real malicious payload.

## Layout
```
src/obfuscation.py    six operators: tags, zero-width, homoglyph, typoglycemia, base64, hybrid
src/defense.py        NFKC + format/tag stripping + homoglyph folding (the one defence)
src/retrieval.py      BM25, Contriever, E5 -> P_ret (Recall@10)
src/guardrail.py      protectai DeBERTa injection classifier -> ε
src/generation.py     Llama-3-8B-Instruct -> P_gen (canary)
src/run_experiment.py runs each stage, writes per-query CSVs to results/<dataset>/
src/make_tables.py    every number for the report as paste-ready LaTeX rows
src/plot_results.py   the four figures -> results/<dataset>/figs/
```

## Install (once)
```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt
```

## Run (per dataset: nq, then hotpotqa)
```
python src/run_experiment.py --dataset nq --stage retrieval   # downloads BEIR once
python src/run_experiment.py --dataset nq --stage evasion
python src/run_experiment.py --dataset nq --stage execution   # optional, GPU
python src/run_experiment.py --dataset nq --stage defence
python src/run_experiment.py --dataset nq --stage combine
python src/make_tables.py nq       # -> results/nq/tables.tex
python src/plot_results.py nq      # -> results/nq/figs/*.pdf
```
Defaults: 200 queries, 5,000 distractor passages, BM25 + Contriever + E5.
Change with `--n_queries`, `--corpus_size`, `--retrievers bm25,e5`.
For a quick smoke test first, add `--n_queries 20 --corpus_size 1000`
(and delete the `results/nq` folder before the real run).

The execution stage needs a GPU with ~16 GB+ memory and access to the gated
Llama-3 model (accept the licence on its HuggingFace page, then
`huggingface-cli login`). Cap it with `--n_exec 100`. If you skip it, the
analysis assumes `P_gen = 1` and reports the `P_ret · ε` trade-off; say so in
the report.

`python src/run_experiment.py --stage retrieval` with no `--dataset` runs the
4-document demo (BM25 only, no downloads) to check the install.
