import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# --- Configuration ---
TRAIN_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\train_data.csv"
TEST_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\test_data.csv"
MODEL_SAVE_PATH = r"c:\Users\lota\Developments\AISys_25_W\term_project\best_model.pth"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Mapping inputs labels to 0-9 index
# Labels in CSV are "1", "2", ... "10"
# We map "1"->0, "2"->1, ..., "10"->9
LABEL_MAP = {str(i): i - 1 for i in range(1, 11)}


class HandLandmarkDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].astype(str).values
        self.features = self.data.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_str = self.labels[idx]
        label = LABEL_MAP[label_str]
        features = torch.tensor(self.features[idx])
        return features, torch.tensor(label, dtype=torch.long)


class HandGestureModel(nn.Module):
    def __init__(self):
        super(HandGestureModel, self).__init__()
        self.layer1 = nn.Linear(63, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)  # 10 classes (1-10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x


def train():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading datasets...")
    train_dataset = HandLandmarkDataset(TRAIN_CSV)
    test_dataset = HandLandmarkDataset(TEST_CSV)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Initialize Model
    model = HandGestureModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # print("  Model saved!")

    print(f"Training finished. Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
