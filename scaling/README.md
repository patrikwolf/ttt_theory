# Scaling Experiments on MNIST and ImageNet

This directory contains end-to-end experiments for scaling test-time training (TTT) across MNIST and ImageNet/CLIP 
settings. It includes scaling experiments over model and dataset size, comparing globally trained models, 
majority-voting, and TTT. The notion of local fine-tuning (LFT) is used equivalently in the codebase to TTT.

### 💻 Environment

Install the project from the repo root:

```bash
pip install -e .
```

- Python packages are specified in `pyproject.toml`.
- Model weights used by experiments are under `scaling/models/saved/` and loaded by scripts in `scaling/models/`.


## 💾 Data

Download or generate the data, then place it according to the corresponding READMEs. See:

- ImageNet/CLIP: instructions in `scaling/data/imagenet/README.md`
- MNIST: instructions in `scaling/data/mnist/README.md`


## ▶️ How to Run

Run each experiment from its directory. Common flags used across scripts:

- `--datetime` — run identifier (e.g., `2023-10-01_12-00-00`). Used in logs/outputs.
- `--shard_id` — selects a hyperparameter shard/setting for sweeps.
- Other flags are specific to the script (see the file for `argparse` definitions).


## 🧪 MNIST: Examples

- Train a global CNN classifier:

```bash
cd scaling/mnist/a_global_classifier
python train_global_model.py --datetime <datetime_string> --shard_id <shard_id>
```
where `<datetime_string>` is a unique identifier for the run (e.g., `2023-10-01_12-00-00`) and `<shard_id>` is an 
integer indicating the choice of hyperparameter shard (see code for more details).

- Evaluate the global CNN reference model:

```bash
cd scaling/mnist/a_global_classifier
python eval_reference.py
```

## 🗂️ Outputs, Logs, and Results

- Many scripts write per-run artifacts keyed by `--datetime` and optionally by `--shard_id`.
- For hyperparameters, see YAML/JSON files under the respective `parameters/` directories.
- All results are saved in the `results` directory (will be generated automatically)
