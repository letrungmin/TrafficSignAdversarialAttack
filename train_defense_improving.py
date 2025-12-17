import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
import time
import os

 
# 1. SETUP
 
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ USING DEVICE: Apple MPS (M4 GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_CLASSES = 43

     
    # 2. DATA
     
    print("\n[INFO] Preparing Data...")
    
    # Train transform
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

    train_set = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=train_transform)
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

     
    # 3. MODEL & ATTACKER
     
    print("[INFO] Building Improved Robust Model...")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Use PGD to train (Eps=8/255)
    atk_train = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7)

     
    # 4. MIXED TRAINING LOOP
     
    print(f"\n[INFO] Starting MIXED TRAINING (Clean + Adv) for {NUM_EPOCHS} epochs...")
    print("-" * 70)

    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # --- Part 1: Clean image study ---
            outputs_clean = model(inputs)
            loss_clean = criterion(outputs_clean, labels)

            # --- PHẦN 2: Attack image study ---
            model.eval() # Switch to eval mode to generate images.
            adv_inputs = atk_train(inputs, labels)
            model.train() # Back to train to study
            
            outputs_adv = model(adv_inputs)
            loss_adv = criterion(outputs_adv, labels)

            # --- Summary: Add 2 LOSS ---
            # Models must be good at both.
            loss = (0.5 * loss_clean) + (0.5 * loss_adv)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate metrics for tracking
            _, pred_clean = torch.max(outputs_clean.data, 1)
            _, pred_adv = torch.max(outputs_adv.data, 1)
            
            total += labels.size(0)
            correct_clean += (pred_clean == labels).sum().item()
            correct_adv += (pred_adv == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f"   > Batch {i+1} | Loss: {loss.item():.4f}")

        acc_clean_train = 100 * correct_clean / total
        acc_adv_train = 100 * correct_adv / total
        
        # --- Validation (Test trên tập Test sạch) ---
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
              f"Train Clean: {acc_clean_train:.1f}% | "
              f"Train Adv: {acc_adv_train:.1f}% | "
              f"VAL CLEAN: {val_acc:.2f}%")
        

        torch.save(model.state_dict(), "gtsrb_resnet18_robust_mixed.pth")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"✅ Training Completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"💾 Saved as 'gtsrb_resnet18_robust_mixed.pth'")

if __name__ == '__main__':
    main()