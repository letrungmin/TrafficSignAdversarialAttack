import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
def main():
    # Setup Device for M4 (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ USING DEVICE: Apple MPS (M4 GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ USING DEVICE: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ USING DEVICE: CPU (Slow)")

    BATCH_SIZE = 64
    NUM_CLASSES = 43
    
    # Create folder to save attack visualization results
    if not os.path.exists('attack_results'):
        os.makedirs('attack_results')

    # ==========================================
    # 2. LOAD DATA (TEST SET ONLY)
    # ==========================================
    print("\n[INFO] Loading Test Data...")
    
    # Define transforms: Resize -> ToTensor -> Normalize
    # Note: No random augmentation is used for testing/attacking
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Load GTSRB Test Set
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ==========================================
    # 3. LOAD TRAINED MODEL (VICTIM MODEL)
    # ==========================================
    print("[INFO] Loading Victim Model (ResNet18)...")
    
    # Initialize model structure (ResNet18)
    model = torchvision.models.resnet18(weights=None) 
    # Modify the last layer to match GTSRB classes (43)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Load weights trained in Phase 1
    # IMPORTANT: Ensure 'gtsrb_resnet18_clean.pth' exists in the current folder
    model_path = "gtsrb_resnet18_clean.pth"
    
    try:
        # Load state dict
        if torch.backends.mps.is_available():
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("✅ Model weights loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: '{model_path}' not found. Please run Phase 1 (train_baseline.py) first!")
        return
    except Exception as e:
        print(f"❌ Load Error: {e}")
        # Fallback for older pytorch versions
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Move model to device
    model = model.to(device)
    
    # IMPORTANT: Set model to evaluation mode. 
    # If this is forgotten, attacks might fail or give wrong results.
    model.eval() 

    # ==========================================
    # 4. DEFINE ATTACKS (FGSM & PGD)
    # ==========================================
    print("[INFO] Initializing Attacks (FGSM & PGD)...")
    
    # Epsilon 8/255 is a standard perturbation budget in adversarial literature
    atk_fgsm = torchattacks.FGSM(model, eps=8/255)
    
    # PGD is a stronger, iterative attack (Steps=10 means it attacks 10 times per image)
    atk_pgd = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)

    # ==========================================
    # 5. ATTACK LOOP & EVALUATION FUNCTION
    # ==========================================
    
    def evaluate_attack(name, attack_method, dataloader):
        """
        Runs the model on the dataloader with the specified attack method.
        Returns accuracy and saves a visualization image.
        """
        print(f"\n⚡ Running Attack: {name} ...")
        correct = 0
        total = 0
        start_time = time.time()
        
        # Flag to ensure we only save visualization for the first batch
        saved_images = False 
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Generate adversarial images
            if attack_method is None:
                # Clean evaluation (No attack)
                adv_images = images
            else:
                # Attack the images
                adv_images = attack_method(images, labels)
            
            # Predict outputs
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update stats
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save visualization for the very first batch
            if not saved_images and attack_method is not None:
                save_visualization(images, adv_images, labels, predicted, name)
                saved_images = True
                
        # Calculate final accuracy
        acc = 100 * correct / total
        elapsed = time.time() - start_time
        print(f"   -> Result: {name} Accuracy = {acc:.2f}% (Time: {elapsed:.1f}s)")
        return acc

    # Helper function to save comparison images
    def save_visualization(clean_imgs, adv_imgs, labels, preds, name):
        # Un-normalize images to display them correctly [0, 1] range
        clean_imgs = clean_imgs.cpu().detach() / 2 + 0.5
        adv_imgs = adv_imgs.cpu().detach() / 2 + 0.5
        
        # Clamp values to ensure valid image range
        clean_imgs = torch.clamp(clean_imgs, 0, 1)
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
        
        plt.figure(figsize=(12, 6))
        for i in range(5): # Show 5 examples
            # Original Clean Image
            plt.subplot(2, 5, i+1)
            plt.imshow(np.transpose(clean_imgs[i], (1, 2, 0)))
            plt.title(f"True: {labels[i].item()}")
            plt.axis('off')
            
            # Attacked (Adversarial) Image
            plt.subplot(2, 5, i+6)
            plt.imshow(np.transpose(adv_imgs[i], (1, 2, 0)))
            plt.title(f"Adv Pred: {preds[i].item()}") # What the model erroneously predicts
            plt.axis('off')
            
        plt.tight_layout()
        # Save to file (using 'name' which must be safe for filenames)
        save_path = f'attack_results/visual_{name}.png'
        plt.savefig(save_path)
        print(f"   (📸 Saved visualization to {save_path})")
        plt.close()

    # ==========================================
    # 6. RUN EXPERIMENTS
    # ==========================================
    print("-" * 50)
    print("STARTING ROBUSTNESS EVALUATION")
    print("-" * 50)
    
    # 1. Clean Accuracy (Baseline Check)
    # This checks if the model is good on normal data
    acc_clean = evaluate_attack("Clean_Data", None, test_loader)
    
    # 2. FGSM Attack
    # We use underscores (_) instead of slashes (/) to avoid file path errors
    acc_fgsm = evaluate_attack("FGSM_eps_8_255", atk_fgsm, test_loader)
    
    # 3. PGD Attack
    acc_pgd = evaluate_attack("PGD_eps_8_255", atk_pgd, test_loader)

    print("-" * 50)
    print("SUMMARY OF PHASE 2 RESULTS:")
    print(f"1. Clean Accuracy: {acc_clean:.2f}% (Expect ~94-95%)")
    print(f"2. FGSM Accuracy:  {acc_fgsm:.2f}%  (Expect significant drop)")
    print(f"3. PGD Accuracy:   {acc_pgd:.2f}%   (Expect near 0%)")
    print("-" * 50)

if __name__ == '__main__':
    main()