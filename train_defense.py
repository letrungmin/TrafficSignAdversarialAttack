import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
import time
import os
import copy

 
# 1. SETUP & CONFIGURATION
 
def main():
    # Setup Device for M4 (Apple Silicon) / CUDA / CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ USING DEVICE: Apple MPS (M4 GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ USING DEVICE: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ USING DEVICE: CPU (Slow)")

    # Hyperparameters
    # Adversarial Training is computationally expensive, so we use 10 epochs
    NUM_EPOCHS = 10          
    BATCH_SIZE = 64          
    LEARNING_RATE = 0.001
    NUM_CLASSES = 43         

     
    # 2. DATA PREPARATION
     
    print("\n[INFO] Preparing Data...")

    # Train transform: Same as Phase 1 (No Horizontal Flip to preserve meaning)
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

    # Load GTSRB Dataset
    train_set = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=train_transform)
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"   - Training Images: {len(train_set)}")
    print(f"   - Testing Images:  {len(test_set)}")

     
    # 3. BUILD MODEL (FRESH RESNET18)
     
    print("\n[INFO] Building Model for Robust Training...")
    
    # Initialize a fresh ResNet18 model
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # DEFINE THE ATTACKER FOR TRAINING (The Sparring Partner)
    # We use PGD for training because it creates the strongest defense.
    # steps=7 is a balance between training speed and robustness.
    atk_train = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7)

     
    # 4. ADVERSARIAL TRAINING LOOP
     
    print(f"\n[INFO] Starting ADVERSARIAL TRAINING for {NUM_EPOCHS} epochs...")
    print("(Note: This will be slower than standard training due to on-the-fly attack generation)")
    print("-" * 70)

    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # --- Training Phase ---
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # ----------------------------------------------------
            # CRITICAL STEP: GENERATE ADVERSARIAL EXAMPLES
            # ----------------------------------------------------
            # Instead of training on clean 'inputs', we generate 'adv_inputs'
            adv_inputs = atk_train(inputs, labels)
            
            optimizer.zero_grad()
            
            # Feed the ATTACKED images into the model
            outputs = model(adv_inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 100 batches
            if (i+1) % 100 == 0:
                print(f"   > Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Calculate Accuracy on Adversarial Examples (How well it learns to resist)
        train_acc = 100 * correct / total
        
        # --- Validation Phase (Test on CLEAN Data) ---
        # We check this to ensure the model still recognizes normal signs correctly
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
              f"Adv Train Acc: {train_acc:.2f}% | "
              f"Clean Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint after every epoch
        torch.save(model.state_dict(), "gtsrb_resnet18_robust.pth")

     
    # 5. FINISH
     
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"✅ Adversarial Training Completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"💾 Robust model saved as 'gtsrb_resnet18_robust.pth'")

if __name__ == '__main__':
    main()