import os
import yaml
import json


def get_name_of_global_linear_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{base_dir}/reference_model.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data['reference_model'], data['cluster']


def get_optimal_mlp_parameters():
    base_dir = os.path.dirname(__file__)
    with open(f'{base_dir}/models_opt_params.json', 'r') as f:
        models = json.load(f)
    return models


if __name__ == '__main__':
    name, cluster = get_name_of_global_linear_model()
    print(f'Global linear model name: {name}')
    print(f'Is cluster: {cluster}')