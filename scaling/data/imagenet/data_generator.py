import torch
import os
import clip

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/16", device=device)
model.eval()

# Transformation
transform = Compose([
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
              std=(0.26862954, 0.26130258, 0.27577711))
])

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = transform(item["image"].convert("RGB"))
        return image, item["label"]


def process_split(split_name, batch_size=64):
    raw_ds = load_dataset("evanarlian/imagenet_1k_resized_256", split=split_name)
    dataset = CLIPImageDataset(raw_ds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    out_dir = f"clip_{split_name}_chunks"
    os.makedirs(out_dir, exist_ok=True)
    batch_count = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc=f"Processing {split_name}")):
            images = images.to(device)

            # Extract & normalize
            emb = model.encode_image(images)                     # [B, 512]
            emb = emb / emb.norm(dim=-1, keepdim=True)           # Normalize to unit norm
            emb = emb.detach().cpu().float()                     # ✅ cast to float32

            labels = labels.cpu()

            # Save
            save_path = os.path.join(out_dir, f"batch_{batch_count:05d}.pt")
            torch.save({
                "embeddings": emb,
                "labels": labels
            }, save_path)

            batch_count += 1
            del images, labels, emb
            torch.cuda.empty_cache()

    print(f"✅ Done: Saved {batch_count} batches for {split_name}")


def load_batched_clip_embeddings(data_dir):
    all_embeddings = []
    all_labels = []

    # Sort to ensure batches are loaded in order
    file_names = sorted(f for f in os.listdir(data_dir) if f.endswith('.pt'))

    for file_name in file_names:
        path = os.path.join(data_dir, file_name)
        data = torch.load(path)

        all_embeddings.append(data["embeddings"])
        all_labels.append(data["labels"])

    # Optionally concatenate to single tensors
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels


if __name__ == "__main__":
    # Define batch size
    batch_size = 64

    # Run both splits
    process_split("val", batch_size=batch_size)
    process_split("train", batch_size=batch_size)
    print('Done!')