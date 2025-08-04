import os
import random
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import timm

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 

# --------------- GLOBAL SEED FOR REPRODUCIBILITY ---------------
SEED = 1
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

def _init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# --------------- DATASET BUILDER ---------------
class Dataset_Builder(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None):
        self.transform = transform
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.all_images = self.fake_images + self.real_images
        self.labels = [1] * len(self.fake_images) + [0] * len(self.real_images)  # 1 for fake, 0 for real

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        name_id = image_path.split('/')[-1].replace('.jpg', '')
        return image, label, name_id

# Function to split dataset into training and validation sets
def split_dataset(dataset, train_size, val_size, seed):
    torch.manual_seed(seed)
    train_data, val_data = random_split(dataset, [train_size, val_size])
    return train_data, val_data

# --------------- MODEL DEFINITION ---------------
class Network(nn.Module):
    def __init__(self, base_model_name="efficientnet_b0"):
        super(Network, self).__init__()

        base_model = timm.create_model(base_model_name, pretrained=True)
        self.features = base_model

        # Freeze initials layers
        for param in self.features.parameters():
            param.requires_grad = False  
            
        # Adjust output layer for binary classification
        num_features = base_model.classifier.in_features
        base_model.classifier = nn.Linear(num_features, 1)  

    def unfreeze_last_layers(self, num_layers=5):
        layers = list(self.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        return x

# --------------- DATA LOADING ---------------

# Set dataset paths
root_path_fake = '/kaggle/input/exp-data/EXP/FAKE'
root_path_real = '/kaggle/input/exp-data/EXP/REAL'

transform = transforms.Compose([
    transforms.Resize((450, 450)),  # input size
    transforms.ToTensor(),
])

# Load dataset
dataset = Dataset_Builder(root_path_fake, root_path_real, transform=transform)

data_size = len(dataset)
train_size = int(0.8 * data_size)
test_size = data_size - train_size
train_dataset, val_dataset = split_dataset(dataset, train_size, test_size, SEED)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, worker_init_fn=_init_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, worker_init_fn=_init_fn)

# --------------- TRAINING CONFIGURATION ---------------
epochs = 10
lr = 0.0001

torch.manual_seed(SEED)
Net = Network(base_model_name="efficientnet_b0")
Net.unfreeze_last_layers(5)  # Unfreeze the last 5 layers
model = Net.to(device)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
loss_fn = nn.BCEWithLogitsLoss()


# Lists for storing training metrics
train_losses = []
val_accuracies = []

# Early Stopping Parameters
patience = 3  # Number of epochs to wait for improvement
best_val_loss = float('inf')
trigger = 0

# --------------- TRAINING LOOP ---------------
for epoch in range(epochs):
    model.train()
    total_loss, corrects = 0, 0

    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, targets = data.to(device), target.float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.round(torch.sigmoid(output))
        corrects += (preds == targets).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * corrects / len(train_loader.dataset)
    train_losses.append(avg_loss)

   # --------------- VALIDATON LOOP ---------------
    model.eval()
    val_loss, corrects = 0, 0
    all_preds = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for data, target, _ in val_loader:
            data, targets = data.to(device), target.float().unsqueeze(1).to(device)

            output = model(data)
            loss = loss_fn(output, targets)
            val_loss += loss.item()
            preds = torch.round(torch.sigmoid(output))
            corrects += (preds == targets).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.sigmoid(output).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100. * corrects / len(val_loader.dataset)
    val_accuracies.append(val_accuracy)

    # Print training progress
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 15)
    print(f"Train Loss: {avg_loss:.4f} train Acc: {accuracy:.4f}%")
    print(f"Validation Loss: {avg_val_loss:.4f} Validation Accuracy: {val_accuracy:.2f}%")

 # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        print("Best model saved.")
    else:
        trigger += 1
        print(f"Early stopping trigger: {trigger}/{patience}")
        if trigger >= patience:
            print("Early stopping activated.")
            break

# ------------ Final Evaluation Metrics ------------------
auc = roc_auc_score(all_targets, all_probs)
tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)

print(f"\nFinal Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")

# ------------ TRAINING LOSS & VALIDATION ACCURACY PLOT ---------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss", color="blue")
plt.tick_params(axis='y', labelcolor="blue")

ax2 = plt.gca().twinx()
ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", marker="o", linestyle="--", color="green")
ax2.set_ylabel("Accuracy (%)", color="green")
ax2.tick_params(axis='y', labelcolor="green")
plt.title("Training Loss and Validation Accuracy Over Epochs")
plt.grid()
plt.legend(loc="upper center")
plt.show()


# ------------ CONFUSION MATRIX HEATMAP ---------------
cm = confusion_matrix(all_targets, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



# ------------ BATCH VISUALIZATION FUNCTION ---------------
def visualize_batch(images, labels, batch_size, title="Batch of Images"):
    plt.figure(figsize=(12, 12))
    for i in range(min(len(images), batch_size)):
        plt.subplot(3, 3, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        plt.imshow(img)
        plt.title("Fake" if labels[i] == 1 else "Real")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Retrieve a batch of images
batch_size = 9
dataiter = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=_init_fn))
images, labels, _ = next(dataiter)

# Move images to CUDA device if available
if torch.cuda.is_available():
    images = images.cuda()

# Visualize the batch of images
visualize_batch(images, labels, batch_size)