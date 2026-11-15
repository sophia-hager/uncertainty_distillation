# Uncertainty Distillation

Code to replicate the uncertainty distillation paper.

A language model already has a usable notion of its own uncertainty: sample it
many times and the spread of its answers tells you how sure it is. Uncertainty
distillation makes that signal explicit. We sample a *teacher* model, calibrate
the resulting predictive distribution against correctness, discretize it into a
handful of confidence bins, and fine-tune a *student* (the same model) to state
the right bin in words alongside its answer. The student ends up saying "very
high" when it is usually right and "low" when it is usually wrong, without any
access to logits at inference time.

The pipeline is four stages:

| Stage | Script | What it does |
| --- | --- | --- |
| 1. Sample | `scripts/sample_teacher.py`, `scripts/sample_teacher_open.py` | Draw many completions per question; count how often each answer occurs |
| 2. Distill | `scripts/build_finetuning_data.py`, `scripts/build_finetuning_data_open.py` | Calibrate those frequencies, bin them, and verbalize them as training targets |
| 3. Train | `scripts/train.py` | Supervised fine-tuning of the student on the verbalized confidences |
| 4. Evaluate | `scripts/predict.py`, `scripts/evaluate.py`, `scripts/evaluate_open.py` | Generate on held-out data; report accuracy per confidence bin and AUROC |

Two settings are supported. **Multiple choice** (MMLU, Social IQa, ARC) counts
answers by letter. **Open ended** (GSM8K) groups free-text samples into semantic
equivalence classes with an NLI model first, since string equality would split
one answer across many surface forms.

The same method also runs against a model you can only reach through an API.
[Gemini pipeline](#gemini-pipeline-api-models) covers that path: the teacher is
sampled with the Gemini API and the student is trained with Vertex AI
supervised tuning.

## Install

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

The code has been run with Llama (`meta-llama/Llama-3.2-3B-Instruct`) and
Mistral (`mistralai/Ministral-8B-Instruct-2410`) models; both are gated on the
Hugging Face Hub, so log in with `huggingface-cli login` first.

If FlashAttention 2 is not installed, pass `--attn_implementation sdpa` to
`scripts/train.py` and `scripts/predict.py`.

## Quick start

An end-to-end run on a small slice of Social IQa, scored on ARC-Challenge:

```bash
run/pipeline_multiple_choice.sh meta-llama/Llama-3.2-3B-Instruct
```

and the free-text equivalent on GSM8K:

```bash
run/pipeline_open.sh mistralai/Ministral-8B-Instruct-2410
```

Both are worked examples: raise `LIMIT`, `NUM_SAMPLES` and the shard count for a
full-scale run.

## Running the stages yourself

**1. Sample the teacher.** This is by far the most expensive stage — 100
generations per question — and it is embarrassingly parallel, so it is sharded.
Each shard writes `samples/<dataset>_<model>/sample_<i>.jsonl`:

```bash
python3 scripts/sample_teacher.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset siqa --num_samples 100 \
    --num_shards 32 --shard_index 1
```

`run/sample_teacher.sh` runs the shards one after another; on a cluster, launch
each `--shard_index` as its own job instead.

**2. Build the fine-tuning data.** The relative frequency of each answer is
mapped through an isotonic regression fit on (frequency, correctness) pairs,
discretized into `--n_bins` bins, and verbalized:

```bash
python3 scripts/build_finetuning_data.py \
    --predictions_path 'samples/siqa_Llama-3.2-3B-Instruct/*.jsonl' \
    --output_path datasets/llama_siqa.jsonl \
    --scheme append_all --num_incorrect 1 \
    --calibration_method isotonic --n_bins 5
```

`--scheme` controls which sampled answers become training examples:

- `append_correct` — only the correct answer, with its calibrated confidence;
- `append_all` — the correct answer plus up to `--num_incorrect` incorrect ones,
  so the student also learns what being wrong feels like;
- `append_stratified` — at most one answer per confidence bin.

`--calibration_method identity` skips recalibration and uses the raw sample
frequency, which is the ablation that shows calibration is doing the work.

**3. Fine-tune the student.**

```bash
accelerate launch scripts/train.py \
    --train_data datasets/llama_siqa.jsonl \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --run_name llama_siqa --num_epochs 1 --learning_rate 3e-5
```

Add `--use_lora --lora_r 8 --lora_alpha 16` for the 8B-class models. Pass
`--report_to wandb --wandb_project <project>` to log to Weights & Biases; by
default nothing is reported anywhere.

**4. Predict and score.**

```bash
python3 scripts/write_dataset.py --dataset arc --split validation \
    --output data/arc_validation.jsonl

python3 scripts/predict.py \
    --model_name_or_path runs/llama_siqa/best \
    --data data/arc_validation.jsonl \
    --output_path runs/llama_siqa/predictions.jsonl

python3 scripts/evaluate.py runs/llama_siqa/predictions.jsonl --num_bins 5
```

`scripts/evaluate.py` prints accuracy within each verbalized confidence bin,
overall accuracy, and the AUROC of the confidence against correctness — the
headline number, since a useful confidence is one that *orders* right from
wrong. For open-ended answers use `scripts/evaluate_open.py`, which scores
equivalence with an LLM judge and expects `OPENAI_API_KEY` in the environment.


## Gemini pipeline (API models)

The method does not need weights — only the ability to sample a model many
times and to fine-tune it. `uncertainty_distillation/gemini/` runs the same
four stages against Gemini on Vertex AI, with three differences:

- **Structured output.** Gemini is asked for JSON with a closed `confidence`
  enum, so the verbalized confidence is parsed from a field rather than from
  tags in free text.
- **Cloud storage.** Datasets, batch job inputs and predictions are JSONL files
  in a GCS bucket, because that is how Vertex AI reads and writes data.
- **Asynchronous stages.** Tuning and batch inference are long-running jobs, so
  an experiment is a resumable record: each invocation advances it as far as it
  can and saves its state under `runs/gemini/experiments.json`.

### Setup

```bash
pip install -r requirements-gemini.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your project>
export UD_GCS_BUCKET=<a GCS bucket you own>    # bucket names are globally unique
```

Authentication is Application Default Credentials; no key is stored in this
repository and none is read from the config.

### Running an experiment

```bash
# start, on dev_set_size examples with online inference
python3 scripts/gemini_experiment.py start --config configs/gemini_mmlu.yaml

# or start a full run, which uses batch inference
python3 scripts/gemini_experiment.py start --config configs/gemini_mmlu.yaml --full

# see where each experiment got to
python3 scripts/gemini_experiment.py list

# advance one whenever its cloud jobs have finished
python3 scripts/gemini_experiment.py continue 0 --full
```

`run/pipeline_gemini.sh` wraps the first two of those as a worked example.

Any field of `uncertainty_distillation/gemini/config.py` can be set in a YAML
file or overridden in dot notation on the command line:

```bash
python3 scripts/gemini_experiment.py start n_generations=32 \
    calibration.method=isotonic_regression augmentation_scheme=STRATIFIED
```

### Stages

1. **`GENERATING_SAMPLES`** — draw `n_generations` structured responses per
   training question from `base_model`. Online for small runs; sharded Vertex
   AI batch jobs for full ones.
2. **`CREATE_TUNING_DATA`** — fit the calibrator on a held-out slice of prompts
   (`calibration.validation_split`), map each answer's sample frequency through
   it, bin it with `index = min(floor(p * n_bins), n_bins - 1)`, and write that
   bin into one of the model's own sampled responses. The teacher's own
   accuracy and AUROC are recorded here as the ceiling distillation is trying
   to transfer.
3. **`TUNING`** — Vertex AI supervised tuning on the augmented set
   (`epochs`, `learning_rate_multiplier`, `adapter_size`).
4. **`INFERENCE`** → **`COMPLETE`** — one prediction per held-out example, then
   accuracy, high-confidence accuracy (the top two bins) and the AUROC of the
   verbalized confidence against correctness. The bin is mapped back to a score
   by its position, `index / (n_bins - 1)`.

### Augmentation schemes

| `augmentation_scheme` | Target the student is trained on |
| --- | --- |
| `CORRECT_ONLY` | The correct answer with its calibrated confidence |
| `CORRECT_PLUS_N` | The above plus `augmentation_scheme_n` wrong answers, each with its own confidence |
| `STRATIFIED` | The above plus one wrong answer per occupied confidence bin |
| `BINNED_ANSWERS` | A single target naming an answer for every bin (empty bins are nulls) |
| `SPARSE_BINNED_ANSWERS` | As above, with empty bins omitted |

Set `experiment_type: BASELINE` to skip straight to evaluating the untuned base
model, which is the comparison the distilled numbers are read against.

## Layout

```
uncertainty_distillation/        library: prompts, parsing, calibration, clustering
uncertainty_distillation/gemini/ the same method against the Gemini API
scripts/                         one script per pipeline stage
run/                             plain-bash drivers and worked end-to-end examples
configs/                         example YAML configs for the Gemini pipeline
data/ samples/ datasets/ runs/   outputs, created on demand (git-ignored)
```

All paths are relative to the repository, and nothing is written outside it.

## Notes

- The number of confidence bins (`--num_bins` / `--n_bins`, one of 2, 5, 7, 10)
  must match across stages 2, 3 and 4: it determines the words the student is
  trained to emit and the vocabulary the evaluator parses.
- `uncertainty_distillation/general_calibration_error.py` is the General
  Calibration Error implementation from the Uncertainty Metrics project
  (Apache 2.0), included unchanged.
- The Gemini pipeline uses its own bin vocabulary (`VERY LOW` … `VERY HIGH`,
  five or seven bins) because the bins are an enum in the structured-output
  schema; the local pipeline verbalizes them in `<confidence>` tags instead.
- Nothing in either pipeline reads or writes outside the repository, apart from
  the GCS bucket you configure for the Gemini path.

## License

Apache License 2.0; see [LICENSE](LICENSE). The vendored General Calibration
Error implementation is Apache 2.0 as well, and keeps its original header.
