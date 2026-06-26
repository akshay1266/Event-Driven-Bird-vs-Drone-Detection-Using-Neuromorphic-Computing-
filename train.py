import os
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

from data_pipeline import get_dataloaders, encode_spikes

# ================= MODEL CONFIG =================
width = 2
conv_filter = (5, 5)
pool = (4, 4)
initial_beta = 0.93


class SpikingCNN(nn.Module):
    def __init__(self, spike_grad):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32 * width, kernel_size=conv_filter, padding="same")
        self.bn1 = nn.BatchNorm2d(32 * width)
        self.lif1 = snn.Leaky(beta=initial_beta, learn_beta=True, learn_threshold=True, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(pool)

        self.conv2 = nn.Conv2d(32 * width, 32 * width, kernel_size=conv_filter, padding="same")
        self.bn2 = nn.BatchNorm2d(32 * width)
        self.lif2 = snn.Leaky(beta=initial_beta, learn_beta=True, learn_threshold=True, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(pool)

        self.conv3 = nn.Conv2d(32 * width, 16 * width, kernel_size=conv_filter, padding="same")
        self.bn3 = nn.BatchNorm2d(16 * width)
        self.lif3 = snn.Leaky(beta=initial_beta, learn_beta=True, learn_threshold=True, spike_grad=spike_grad)
        self.pool3 = nn.MaxPool2d(pool)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * width * 4, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.lif_fc1 = snn.Leaky(beta=initial_beta, learn_beta=True, learn_threshold=True, spike_grad=spike_grad)

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 2)
        self.lif_fc2 = snn.Leaky(beta=initial_beta, learn_beta=True, learn_threshold=True, spike_grad=spike_grad)

    def forward(self, x_time):
        utils.reset(self)
        spk_rec = []
        mem_rec = []

        for step in range(x_time.size(0)):
            x = x_time[step]

            x = self.conv1(x)
            x = self.bn1(x)
            x, _ = self.lif1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x, _ = self.lif2(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x, _ = self.lif3(x)
            x = self.pool3(x)

            x = self.flatten(x)
            x = self.fc1(x)
            x = self.bn_fc1(x)
            x, _ = self.lif_fc1(x)

            x = self.dropout(x)

            x = self.fc2(x)
            spk_out, mem_out = self.lif_fc2(x)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# ================= ACCURACY FUNCTION =================
def evaluate_accuracy(net, dataloader, device, num_steps, encoding_type):
    net.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            spike_data = encode_spikes(data, num_steps=num_steps, encoding_type=encoding_type)
            spk_rec, _ = net(spike_data)

            correct += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return correct / total


# ================= TRAIN FUNCTION =================
def train():
    print("Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = r"d:\bird_vs_ drone\birdvdrone\BirdVsDrone"
    num_steps = 30
    batch_size = 16
    epochs = 50
    encoding_type = "rate"

    # ⚠️ IMPORTANT: num_workers = 0 for Windows
    train_loader, test_loader = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0
    )

    spike_grad = surrogate.fast_sigmoid(slope=15)
    net = SpikingCNN(spike_grad=spike_grad).to(device)

    loss_fn = SF.ce_rate_loss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    val_acc_history = []  # ⭐ IMPORTANT

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            spike_data = encode_spikes(data, num_steps=num_steps, encoding_type=encoding_type)
            spk_rec, mem_rec = net(spike_data)

            loss = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = train_loss / len(train_loader)

        # ===== VALIDATION =====
        val_acc = evaluate_accuracy(net, test_loader, device, num_steps, encoding_type)
        val_acc_history.append(val_acc * 100)

        print(f"Validation Accuracy: {val_acc*100:.2f}%")

        # ===== SAVE BEST MODEL =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), "best_snn_model.pth")
            print(f"✔ New best model saved: {best_val_acc*100:.2f}%")

    # ===== SAVE ACCURACY HISTORY =====
    torch.save(val_acc_history, "val_acc_history.pth")
    print("\nTraining complete. Accuracy history saved.")


# ================= MAIN =================
if __name__ == "__main__":
    train()
