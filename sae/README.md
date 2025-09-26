# Sparse Autoencoder for ImageNet-1k CLIP

We provide the implementation of the top-k sparse autoencoder (SAE) model in `sae_topk.py`, which was used in the ImageNet-1k experiments. The SAE was trained on the corresponding CLIP <CLS> token embeddings. The associated model parameters are available in `clip_sae_checkpoint.pt`.

## Extracting CLIP

In order to ensure the consistency of CLIP embeddings, extract them using the accompanying script:
```bash
python extract_clip_embd.py
```
which extracts and **normalizes** the corresponding \<CLS\> token emebedding of dimension 512.

## Usage of Trained Model

To play with trained top-k sparse autoencoder, use the following snippet:
```python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
checkpoint = torch.load('clip_sae_checkpoint.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']

# Initialize the TopKSAE model with parameters from the config
sae = TopKSAE(
    input_dim=config['feature_dim'],
    hidden_dim=config['feature_dim'] * config['expansion_factor'],
    k=config['k'],
    use_ghost_grads=config.get('use_ghost_grads', True),
    ghost_threshold=config.get('ghost_threshold', 0.0),
    ghost_weight=config.get('ghost_weight', 0.01),
    normalize_decoder=config.get('normalize_decoder', True),
    decoder_bias=config.get('decoder_bias', True),
)

# Load the model state from the checkpoint
sae.load_state_dict(checkpoint['model_state_dict'])
sae.to(device)
sae.eval()

# Example input tensor
input_dim = config['feature_dim']
input_tensor = torch.randn(1, input_dim).to(device)
output_tensor = sae(input_tensor)

print(f'Reconstructed output shape: {output_tensor["reconstruction"].shape}')
print(f'Activations shape: {output_tensor["activated"].shape}')
print(f'Pre-activations shape: {output_tensor["pre_activation"].shape}')
print(f'Active mask shape: {output_tensor["active_mask"].shape}')
print(f'Ghost loss: {output_tensor["ghost_loss"].item()}')
```

We provide the config file with training schedule for additional convenience in `config.json`.
