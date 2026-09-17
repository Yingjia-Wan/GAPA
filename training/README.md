# §6.1 — Training

Two stages live here, and they differ in both their data and their split. That is the whole reason they sit in separate directories:

| | trains on | in-domain test | out-of-domain test | split | seeds |
|---|---|---|---|---|---|
| `hp_search/` | `llm`, 104 attributes | `llm`, 42 | `human` (50), `novel` (96) | 60 / 15 / 25, three-way | 42, 123, 456 |
| `predictor/` | `merged`, 221 attributes across all three sources | `merged`, 94 | — | 70 / 0 / 30, two-way | 42, 123, 456 |

Attribute counts, not row counts; splits are grouped by attribute so none straddles two of them. The hyperparameter search never sees `human` or `novel`, which is what makes them out-of-domain for it; the final predictor draws its training and test attributes from all three sources, and `predictor/report_stratified_eval.py` breaks its metrics down by source.

The scripts both stages drive stay at this top level: `prep_data_for_training.py` builds the prompt-formatted splits, `lora_training.py` trains one run, `lora_eval.py` evaluates one, `run_experiments.py` orchestrates a sweep over the grid in an experiment config, and `aggregate_run_seeds.py` pools a run's seeds.

## How a config finds its split

Every experiment config names its own `input_data_base`, `val_size` and `test_size`, and `run_experiments.py` passes them down to data prep. Nothing reads split proportions out of the repo-root `config.json` during a sweep. This matters: the two stages want different splits, so a single global setting meant that running either stage silently relabelled the other stage's prepared data. Keeping the split next to the data it describes is what removes that hazard.

If you invoke `prep_data_for_training.py` by hand, pass `--val_size` / `--test_size` yourself. Held-out evaluation sets take `--force_split_label test` instead, since every row of `human` and `novel` is test.

## The pipeline, in order

1. **`hp_search/configs/model-specific_hp_search.json`** — searches a per-model hyperparameter range and picks the best configuration for each backbone. `hp_search/select_best_runs.py` reads the sweep output and writes the winners.
2. **`hp_search/configs/best_runs_experiment.json`** — retrains each model's winning configuration across three seeds, for a comparison that does not hang on one draw.
3. **`predictor/configs/further_search.json`** — takes the strongest models forward and searches again, now on the *final* training data (`merged`) and the *final* split (70/30). The best run of this stage **is** the GAPA predictor.
4. **`predictor/configs/eval_only.json`** — re-evaluates trained runs without retraining.

`hp_search/configs/global_hp_search.json` is a broader, far more expensive alternative to step 1 that searches one grid across all models. **It was not used for the paper**; it is kept because it is the natural thing to reach for if you are starting from scratch with a different backbone.

## The released model

`predictor/config.json` is the recipe that produced the published predictor: OLMo-2 7B base, LoRA r=16 / alpha=16, lr 1e-4, batch size 8, on `merged` with **seed 123** and a two-way 70/30 split. Its run — metrics, predictions, stratified evaluation — is kept under `predictor/avg_olmo2_7b_base_direct_bs8_lr1e-4_alpha16_r16/`. Weights are not in git; they are on [Hugging Face](https://huggingface.co/alisa-yingjia-wan/gapa-predictor-olmo2-7b).

To rebuild its training data exactly:

```bash
python training/prep_data_for_training.py --input data/merged_clean.csv \
    --output data/merged --extract_type avg --prompt_name direct \
    --seed 123 --val_size 0.0 --test_size 0.3
```

That yields 663 train / 282 test rows over 945 attribute–gender pairs, matching the `predictions.csv` in the run directory pair for pair.

The ablation design and hyperparameter grid are documented in [`hp_search/ABLATIONS.md`](hp_search/ABLATIONS.md).
