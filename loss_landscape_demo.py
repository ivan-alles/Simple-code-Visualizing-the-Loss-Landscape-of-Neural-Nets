""" Demonstrates how to use loss_landscape.py.

    1. Fine-tune a pretrainted on ImageNet model on CIFAR-10 for a few epochs and builds loss landscape.
    2. Evaluate the model on CIFAR-10 test set.
    3. Compute loss landscape and plot it.
"""

from pathlib import Path
import time, os, torch, torchvision as tv
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import loss_landscape

# -----------------------------------------------------------------------------#
CKPT_PATH   = Path("resnet18_cifar10.pt")   # where we keep the fine-tuned model
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32 = True  # faster on Ampere+ GPUs
torch.backends.cudnn.benchmark = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_model() -> nn.Module:
    # 1) instantiate ImageNet-pretrained ResNet-18
    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    # 2) replace final layer with 10-class layer
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, 10)
    # 3) freeze all but the final layer
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    return model.to(DEVICE)

# -----------------------------------------------------------------------------#
# ------------- training & evaluation -----------------------------------------#
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total   = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return 100. * correct / total

def train(model: nn.Module, loader: DataLoader):
    epochs = 3
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.fc.parameters(), lr=1e-3)
    for epoch in range(1, epochs):
        model.train()
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"[{epoch:02}/{epochs}]  loss: {avg_loss:.4f}")

def main():
    if CKPT_PATH.exists():
        print(f"→ Loading checkpoint from {CKPT_PATH} …")
        model = build_model()  # build same architecture
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    else:
        train_tfms = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        train_set = tv.datasets.CIFAR10(root="data", train=True,
                                        transform=train_tfms, download=True)

        train_loader = DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        print("→ No checkpoint found – starting fine-tuning …")
        model = build_model()
        t0 = time.time()
        train(model, train_loader)
        print(f"Training finished in {(time.time() - t0)/60:.1f} min")

        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"✓ Checkpoint saved to {CKPT_PATH}")

    test_tfms = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    test_set = tv.datasets.CIFAR10(
        root="data",
        train=False,
        transform=test_tfms,
        download=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_acc = evaluate(model, test_loader)
    print(f"Top-1 accuracy on CIFAR-10 test set: {test_acc:.2f}%")

    ll, x, y = loss_landscape.compute_loss_landscape(
        model,
        loss_fn=nn.CrossEntropyLoss(),
        device=DEVICE,
        data_loader=test_loader,
    )

if __name__ == "__main__":
    main()
