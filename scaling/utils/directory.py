import os
from pathlib import Path
from datetime import datetime


def get_results_dir(experiment_name, timestamp=None, create_dir=True) -> Path:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(f'{base_dir}/../results/' + experiment_name)
    if timestamp:
        results_dir = Path(f'{output_dir}/{timestamp}')
    else:
        results_dir = Path(f'{output_dir}/{datetime.now().strftime("%Y-%m-%d__%H-%M-%S")}')
    if create_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_models_dir(hydra=False) -> Path:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if hydra:
        models_dir = Path(f'{base_dir}/../models/saved/hydra')
    else:
        models_dir = Path(f'{base_dir}/../models/saved')
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir