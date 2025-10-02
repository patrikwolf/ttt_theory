import torch

from sae.sae_topk import TopKSAE

if __name__ == '__main__':
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model from Hugging Face
    sae = TopKSAE.from_pretrained('patrikwolf/clip-topk-sae')

    # Example: random input vector
    input_dim = sae.input_dim
    clip_embedding = torch.randn(1, input_dim).to(device)

    # Forward pass
    with torch.no_grad():
        output = sae(clip_embedding)

    # Access outputs
    reconstruction = output["reconstruction"]  # Reconstructed embedding
    activations = output["activated"]  # Sparse latent activations
    pre_activations = output["pre_activation"]  # Pre-activation values
    active_mask = output["active_mask"]  # Binary mask of active neurons
    ghost_loss = output["ghost_loss"]  # Auxiliary loss term

    print(f'Reconstructed output shape: {reconstruction.shape}')
    print(f'Activations shape: {activations.shape}')
    print(f'Pre-activations shape: {pre_activations.shape}')
    print(f'Active mask shape: {active_mask.shape}')
    print(f'Ghost loss: {ghost_loss.item()}')