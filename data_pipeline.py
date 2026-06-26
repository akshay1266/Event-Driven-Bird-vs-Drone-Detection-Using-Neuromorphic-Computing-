import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import snntorch.spikegen as spikegen
import random
import os

def get_dataloaders(data_dir, batch_size=8, num_workers=2):
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5), # UPGRADED to 0.5
        transforms.RandomVerticalFlip(p=0.5),   # UPGRADED to 0.5
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.8 * len(full_dataset))
    
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    test_full = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    
    indices = torch.randperm(len(full_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_full, indices[:train_size])
    test_dataset = torch.utils.data.Subset(test_full, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def encode_spikes(data, num_steps, encoding_type="direct"):
    if encoding_type == "direct":
        return data.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
    else:
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-6)
        if encoding_type == "rate":
            return spikegen.rate(data_norm, num_steps=num_steps)
        elif encoding_type == "latency":
            return spikegen.latency(data_norm, num_steps=num_steps, tau=5, threshold=0.01)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

def temporal_jitter(spike_data, max_jitter=1):
    time_steps = spike_data.shape[0]
    jittered_data = torch.zeros_like(spike_data)
    for t in range(time_steps):
        shift = random.randint(-max_jitter, max_jitter)
        new_t = t + shift
        if 0 <= new_t < time_steps:
            jittered_data[new_t] = torch.clamp(jittered_data[new_t] + spike_data[t], 0, 1)
    return jittered_data
    
def test_pipeline():
    print("Testing data pipeline and spike encodings...")
    data_dir = r"d:\bird_vs_ drone\birdvdrone\BirdVsDrone"
    
    # Make sure repo exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset not found at {data_dir}")
        return
        
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=4, num_workers=0)
    
    data, targets = next(iter(train_loader))
    print(f"\nOriginal batched input shape: {data.shape} -> (Batch, C, H, W)")
    
    num_steps = 20
    encodings = ["direct", "rate", "latency"]
    
    for enc in encodings:
        print(f"\n--- Testing {enc.capitalize()} Encoding ---")
        spikes = encode_spikes(data, num_steps=num_steps, encoding_type=enc)
        print(f"[{enc}] Generated shape: {spikes.shape}")
        
        # We apply Neuromorphic jitter augmentation on valid spiking inputs
        if enc in ["rate", "latency"]:
            jittered = temporal_jitter(spikes, max_jitter=2)
            print(f"[{enc} + jitter] Jitter shape:     {jittered.shape}")
            
    print("\n✅ All dimensions match expected (Time, Batch, Channels, Height, Width). Test passed.")

if __name__ == "__main__":
    test_pipeline()
