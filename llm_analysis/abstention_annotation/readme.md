# Abstention annotation sampling

Sampling uses source datasets only (`combined_llm`, `eval_novel`, `eval_human`) and dedupes by response so pooled `all` rows are never double-counted.

## Abstention counts (source datasets only)

**197** unique abstention responses across all models (12,211 non-abstention responses in the same pool).

### By gender

| person_term | Count |
|-------------|------:|
| woman | 23 |
| man | 15 |
| nonbinary person | 159 |
| **Total** | **197** |

### By model

| model | Count |
|-------|------:|
| Qwen2.5-14B-Instruct | 48 |
| Qwen2.5-32B-Instruct | 35 |
| gemini-3-flash-preview | 33 |
| Qwen2.5-7B-Instruct | 27 |
| Meta-Llama-3-8B-Instruct | 26 |
| Mistral-7B-Instruct-v0.3 | 24 |
| Qwen2.5-3B-Instruct | 4 |
| All other models | 0 |
| **Total** | **197** |

Only 7 models produced any abstentions; all base/non-instruct and proprietary models had zero.

### By model × gender

| model | woman | man | nonbinary person | total |
|-------|------:|----:|-----------------:|------:|
| Qwen2.5-14B-Instruct | 13 | 7 | 28 | 48 |
| Qwen2.5-32B-Instruct | 0 | 3 | 32 | 35 |
| gemini-3-flash-preview | 2 | 0 | 31 | 33 |
| Qwen2.5-7B-Instruct | 2 | 1 | 24 | 27 |
| Meta-Llama-3-8B-Instruct | 5 | 3 | 18 | 26 |
| Mistral-7B-Instruct-v0.3 | 1 | 1 | 22 | 24 |
| Qwen2.5-3B-Instruct | 0 | 0 | 4 | 4 |
| **Total** | **23** | **15** | **159** | **197** |

## Sample sizes

| | Count |
|--|------:|
| Unique abstentions (population) | 197 |
| Abstention sample (50%) | 98 |
| Non-abstention controls (matched) | 98 |
| Total for annotation | 196 |

## Usage

```bash
cd GAPA/llm_analysis/abstention_annotation

python3 sample_abstention.py          # 50% of unique abstentions → abstention_sample.csv
python3 sample_nonabstention.py       # matched non-abstention controls + annotation files
python3 eval_annotation.py            # full: both annotators + inter-rater agreement
python3 eval_annotation.py --mode simple   # single annotator (default: annotator1)
python3 eval_annotation.py --mode simple --annotator 2
```
