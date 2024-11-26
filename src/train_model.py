from dataset_loader import TurtleDataset
from unet_model import get_unet
from torch.utils.data import DataLoader
import torch

# Paths to dataset
train_images = "../dataset/train/images/"
train_masks = "../dataset/train/masks/"
val_images = "../dataset/val/images/"
val_masks = "../dataset/val/masks/"

# Create datasets and dataloaders
train_dataset = TurtleDataset(train_images, train_masks)
val_dataset = TurtleDataset(val_images, val_masks)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load model
model = get_unet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}")
