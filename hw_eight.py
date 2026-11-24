
# ==============================================
# ML Zoomcamp 2026 - Homework 8
# Neural Networks and Deep Learning
# ==============================================

import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# ==============================================
# 1. SETUP
# ==============================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(2)

device = torch.device("cpu")

train_path = "data/data/train"
test_path = "data/data/test"


# ==============================================
# 2. DATA PREPARATION
# ==============================================

train_transform = transforms.Compose([
    transforms.Resize(
        (200, 200),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize(
        (200, 200),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ImageFolder(
    train_path,
    transform=train_transform
)

test_dataset = ImageFolder(
    test_path,
    transform=test_transform
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))
print("Class mapping:", train_dataset.class_to_idx)

train_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=0,
    generator=train_generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=20,
    shuffle=False,
    num_workers=0
)


# ==============================================
# 3. MODEL
# ==============================================

class HairModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            3, 32,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.hidden = nn.Linear(32 * 99 * 99, 64)

        self.output = nn.Linear(64, 1)

    def forward(self, x):

        x = self.pool(F.relu(self.conv(x)))

        x = x.flatten(start_dim=1)

        x = F.relu(self.hidden(x))

        return self.output(x)


model = HairModel().to(device)


# ==============================================
# QUESTION 1: LOSS FUNCTION
# ==============================================

criterion = nn.BCEWithLogitsLoss()

print("\nQ1: Loss function =", criterion)


# ==============================================
# QUESTION 2: NUMBER OF PARAMETERS
# ==============================================

total_params = sum(
    p.numel() for p in model.parameters()
)

print("\nQ2: Total parameters =", total_params)


# ==============================================
# 4. OPTIMIZER
# ==============================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.002,
    momentum=0.8
)


# ==============================================
# 5. EVALUATION FUNCTION
# ==============================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_loss += loss.item() * images.size(0)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += images.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


# ==============================================
# 6. TRAINING FUNCTION
# ==============================================

def train_model(model, train_loader, test_loader,
                optimizer, epochs):

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "evaluation_loss": [],
        "evaluation_accuracy": []
    }

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_correct = 0
        total_examples = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_loss += loss.item() * images.size(0)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += images.size(0)

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples

        eval_loss, eval_accuracy = evaluate(
            model, test_loader
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["evaluation_loss"].append(eval_loss)
        history["evaluation_accuracy"].append(eval_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Eval Accuracy: {eval_accuracy:.4f}"
        )

    return history


# ==============================================
# QUESTION 3: BASELINE TRAINING
# ==============================================

print("\n========== BASELINE TRAINING ==========")

history_baseline = train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    epochs=10
)

os.makedirs("runs/reference", exist_ok=True)

with open("runs/reference/history_baseline.json", "w") as f:
    json.dump(history_baseline, f, indent=4)


# Q3: Median training accuracy

q3 = np.median(
    history_baseline["train_accuracy"]
)

print("\nQ3: Median training accuracy =", round(q3, 2))


# ==============================================
# QUESTION 4: STANDARD DEVIATION OF TRAIN LOSS
# ==============================================

q4 = np.std(
    history_baseline["train_loss"],
    ddof=0
)

print("\nQ4: Standard deviation of train loss =", round(q4, 3))


# ==============================================
# 7. DATA AUGMENTATION
# ==============================================

augmented_transform = transforms.Compose([
    transforms.Resize(
        (200, 200),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),

    transforms.RandomRotation(
        50,
        interpolation=transforms.InterpolationMode.NEAREST
    ),

    transforms.RandomResizedCrop(
        200,
        scale=(0.9, 1.0),
        ratio=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================================
# 8. AUGMENTED DATASET
# ==============================================

augmented_dataset = ImageFolder(
    train_path,
    transform=augmented_transform
)

augmented_generator = torch.Generator().manual_seed(SEED + 1)

augmented_loader = DataLoader(
    augmented_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=0,
    generator=augmented_generator
)


# ==============================================
# QUESTION 5 & 6: AUGMENTED TRAINING
# ==============================================

print("\n========== AUGMENTED TRAINING ==========")

# Continue with SAME model and SAME optimizer

history_augmented = train_model(
    model,
    augmented_loader,
    test_loader,
    optimizer,
    epochs=10
)

with open("runs/reference/history_augmented.json", "w") as f:
    json.dump(history_augmented, f, indent=4)


# ==============================================
# QUESTION 5: MEAN EVALUATION LOSS
# ==============================================

q5 = np.mean(
    history_augmented["evaluation_loss"]
)

print("\nQ5: Mean evaluation loss =", round(q5, 3))


# ==============================================
# QUESTION 6: LAST 5 EVALUATION ACCURACIES
# ==============================================

q6 = np.mean(
    history_augmented["evaluation_accuracy"][-5:]
)

print("\nQ6: Mean last 5 evaluation accuracies =", round(q6, 2))


# ==============================================
# FINAL RESULTS
# ==============================================

print("\n========== FINAL ANSWERS ==========")

print("Q1: BCEWithLogitsLoss()")
print("Q2:", total_params)
print("Q3:", round(q3, 2))
print("Q4:", round(q4, 3))
print("Q5:", round(q5, 3))
print("Q6:", round(q6, 2))
