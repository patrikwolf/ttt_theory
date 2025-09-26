import os
import yaml
import json


def get_name_of_global_mnist_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{base_dir}/reference_mnist_model.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data['reference_model'], data['size_scale'], data['cluster']


def get_name_of_mnist_linear_head():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{base_dir}/linear_head.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data['linear_head'], data['cluster']


def get_optimal_mnist_cnn_parameters():
    base_dir = os.path.dirname(__file__)
    with open(f'{base_dir}/mnist_cnn_opt_params.json', 'r') as f:
        models = json.load(f)
    return models


def get_optimal_lft_cnn_parameters():
    base_dir = os.path.dirname(__file__)
    with open(f'{base_dir}/lft_cnn_opt_params.json', 'r') as f:
        models = json.load(f)
    return models


def get_optimal_moe_parameters():
    base_dir = os.path.dirname(__file__)
    with open(f'{base_dir}/moe_opt_params.json', 'r') as f:
        models = json.load(f)
    return models


if __name__ == '__main__':
    name, size_scale, cluster = get_name_of_global_mnist_model()
    print(f'Global linear model name: {name}')
    print(f'Is cluster: {cluster}')