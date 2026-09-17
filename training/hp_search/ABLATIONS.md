# Ablations

## Factors of combination

1. **Data Type** (2 options)
   - `raw`: Individual ratings per participant
   - `avg`: Averaged ratings per attribute-person pair

2. **Model Type** (multiple options, each with base/instruct variants)
   - `llama3_8b_instruct` / `llama3_8b_base`: Llama 3 8B
   - `llama31_8b_instruct` / `llama31_8b_base`: Llama 3.1 8B
   - `phi3_small_instruct` / `phi3_small_base`: Phi-3 Small
   - `mistral_small_instruct` / `mistral_small_base`: Mistral Small
   - `vicuna_13b_instruct` / `vicuna_13b_base`: Vicuna 13B
   - `qwen25_14b_instruct` / `qwen25_14b_base`: Qwen2.5 14B
   - `qwen25_32b_instruct` / `qwen25_32b_base`: Qwen2.5 32B

3. **Prompt Type**
   see prompts.

## Configuration

### Training configuration

Base training parameters are in `config.json` (primary source). Each experiment inherits these and adds:
- `model_name`: Model to use
- `experiment_name`: Unique identifier
- `results_dir`: Where to save results
- `metadata`: Experiment description


## Interpreting results

### Metrics

- **RMSE (Root Mean Squared Error)**: Primary metric, lower is better
  - Penalizes large errors more than small ones
  - Scale: same as original ratings (1-7)
  
- **MAE (Mean Absolute Error)**: Secondary metric, lower is better
  - Average absolute difference between predictions and labels
  - More interpretable than RMSE

- **Per-Gender Metrics**: Evaluate fairness
  - Check if performance is balanced across genders
  - Identify biases in model predictions


## Experiment commands

## Common commands

### Prepare all experiments (no training)
```bash
# Default input data base from experiments.json (train_df_clean)
python training/run_experiments.py

# Custom input data base
python training/run_experiments.py --input-data data/train_df_clean.csv

# Specific experiments
python training/run_experiments.py --experiments \
    avg_llama3_8b_instruct_vanilla \
    avg_llama3_8b_base_vanilla \
    avg_llama3_8b_instruct_direct \
    avg_llama3_8b_base_direct
```

### Run all experiments with training
```bash
# Default input data base
python training/run_experiments.py --full
```

### Run specific experiments
```bash
# Single experiment
python training/run_experiments.py --full --experiments avg_llama3_8b_instruct_vanilla
python training/run_experiments.py --full --experiments avg_gpt2_xl_vanilla
python training/run_experiments.py --full --experiments avg_llama3_8b_instruct_vanilla avg_llama3_8b_base_direct

# Multiple experiments with custom input data
python training/run_experiments.py --full --input-data data/train_df_clean.csv --experiments \
    avg_llama3_8b_instruct_vanilla \
    avg_llama3_8b_base_vanilla \
    avg_llama3_8b_instruct_direct \
    avg_llama3_8b_base_direct

python training/run_experiments.py --full --experiments \
    avg_llama3_8b_instruct_direct \
    avg_llama3_8b_base_direct
```

## Experiment naming convention

Format: `{data_type}_{model_type}_{prompt_name}`

Examples:
- `avg_llama3_8b_instruct_vanilla` - Averaged data, Llama 3 8B instruct model, vanilla prompt
- `raw_llama3_8b_base_default` - Raw data, Llama 3 8B base model, default prompt
- `avg_qwen25_14b_instruct_detailed` - Averaged data, Qwen2.5 14B instruct model, detailed prompt


## Running a single experiment manually

See README.md

## Useful filters

### List all experiments
```bash
ls results/
```

### Find completed experiments
```bash
find results -name "metrics.csv" -type f
```


### Quick performance check
```bash
# Show final RMSE for each experiment
for exp in results/*/run_*/metrics.csv; do
    echo "$exp:"
    tail -1 "$exp" | cut -d',' -f1,11
done
```

### Run overnight batch
```bash
# Prepare first
python training/run_experiments.py

# Review experiment plan
cat experiment_log_*.csv

# Run full suite (takes time!)
nohup python training/run_experiments.py --full > training.log 2>&1 &

# Monitor progress
tail -f training.log
```


### Plot with Python

```bash
python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results/avg_llama3_8b_instruct_vanilla/run_*/metrics.csv')
df.plot(x='step', y='eval_loss')
plt.savefig('temp_plot.png')
"
```
