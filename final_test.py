import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
import os
import time

# 1. SETUP
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ USING DEVICE: Apple MPS ")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    BATCH_SIZE = 64
    NUM_CLASSES = 43

    # 2. LOAD TEST DATA
    print("\n[INFO] Loading Test Data...")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. LOAD ROBUST MODEL (MIXED VERSION)
    print("[INFO] Loading ROBUST Model (Mixed Phase 3)...")
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    model_path = "gtsrb_resnet18_robust_mixed.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ Model '{model_path}' loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: '{model_path}' not found.")
        return

    model = model.to(device)
    model.eval()

    # 4. DEFINE ATTACKS
    atk_fgsm = torchattacks.FGSM(model, eps=8/255)
    atk_pgd = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)

    # 5. EVALUATION FUNCTION
    def test_model(name, attack_method):
        correct = 0
        total = 0
        start = time.time()
        print(f"   Testing {name}...", end="\r")
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if attack_method:
                images = attack_method(images, labels)
                
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc = 100 * correct / total
        print(f"   -> Result: {name} Accuracy = {acc:.2f}% (Time: {time.time()-start:.1f}s)")
        return acc

    # 6. RUN FINAL BENCHMARK    
    print("-" * 60)
    print("🚀 FINAL RESULTS: ROBUST MODEL (MIXED TRAINING)")
    print("-" * 60)

    # 1. Test on Clean Data
    acc_clean = test_model("Clean Data", None)

    # 2. Test against FGSM
    acc_fgsm = test_model("FGSM Attack", atk_fgsm)

    # 3. Test against PGD
    acc_pgd = test_model("PGD Attack ", atk_pgd)
    
    print("-" * 60)
    print("SUMMARY:")
    print(f"1. Clean Accuracy: {acc_clean:.2f}% (Expect > 60%)")
    print(f"2. FGSM Accuracy:  {acc_fgsm:.2f}%  (Expect > 50%)")
    print(f"3. PGD Accuracy:   {acc_pgd:.2f}%   (Expect > 45%)")
    print("-" * 60)

if __name__ == '__main__':
    main()