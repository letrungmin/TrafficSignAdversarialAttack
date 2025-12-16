import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import copy

# ==========================================
# 1. CONFIGURATION
# ==========================================
def main():
    # Check for Apple Silicon GPU (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ USING DEVICE: Apple MPS ")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ USING DEVICE: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ USING DEVICE: CPU (Slow)")

    # Hyperparameters
    NUM_EPOCHS = 20          
    BATCH_SIZE = 64          
    LEARNING_RATE = 0.001
    NUM_CLASSES = 43         

    # 2. DATA PREPARATION
    print("\n[INFO] Preparing Data...")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),         
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Load Datasets
    train_set = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=train_transform)
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"   - Training Images: {len(train_set)}")
    print(f"   - Testing Images:  {len(test_set)}")

    # 3. BUILD MODEL (RESNET18)
    print("\n[INFO] Building ResNet18 Model...")

    # Load pre-trained ResNet18
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    # Modify the last fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # Move model to GPU
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. TRAINING LOOP
    print(f"\n[INFO] Starting Training for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0
        
        # --- Training Phase ---
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 

            optimizer.zero_grad()           
            outputs = model(inputs)         
            loss = criterion(outputs, labels)
            loss.backward()                 
            optimizer.step()                

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        
        # --- Validation Phase ---
        model.eval()  
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): 
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "gtsrb_resnet18_clean.pth") 

    # 5. FINISH
    time_elapsed = time.time() - start_time
    print("-" * 60)
    print(f"✅ Training Completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"💾 Best model saved as 'gtsrb_resnet18_clean.pth'")

if __name__ == '__main__':
    main()