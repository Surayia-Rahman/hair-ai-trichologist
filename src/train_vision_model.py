import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Import configurations directly from our vision classifier module
from vision_classifier import get_data_transformers, build_model

def train_hair_classifier(data_dir, output_model_path, figures_dir, epochs=3, batch_size=32):
    print("=" * 60)
    print("          STARTING AI HAIR CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Set device engine (Lock onto Colab's T4 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Execution Device: {device.type.upper()}")
    
    # 1. Prepare Datasets and DataLoaders
    transform = get_data_transformers()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    
    # Save class names as a text file for our live UI inference engine to read later
    with open("hair-ai-trichologist/src/classes.txt", "w") as f:
        f.write("\n".join(full_dataset.classes))
    
    # Split into 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f" -> Total Images Found: {len(full_dataset)} across {num_classes} categories.")
    print(f" -> Training Allocation: {len(train_dataset)} | Validation Allocation: {len(val_dataset)}")
    
    # 2. Instantiate Model, Loss Function, and Optimizer
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters of our custom classification head (since the base is frozen)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Tracking logs for plotting
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    # 3. Core Training Loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        corrects = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = corrects.double() / len(val_dataset)
        
        # Save historical metrics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc.item())
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({duration:.1f}s) -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
    # 4. Save Trained Model Weights
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"\n[SUCCESS] Model weights saved securely to: {output_model_path}")
    
    # 5. Plot and Export Training Diagnostics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history["train_loss"], label="Train Loss", marker='o')
    plt.plot(range(1, epochs+1), history["val_loss"], label="Val Loss", marker='s')
    plt.title("Model Cross-Entropy Loss Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history["val_acc"], label="Validation Accuracy", color="green", marker='^')
    plt.title("Model Validation Accuracy Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    curve_plot_path = os.path.join(figures_dir, "training_metrics_curves.png")
    plt.savefig(curve_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Training convergence visual graphics saved to: {curve_plot_path}\n")

if __name__ == "__main__":
    RAW_DIR = "hair-ai-trichologist/data/raw/hair_images"
    MODEL_OUT = "hair-ai-trichologist/data/processed/efficientnet_hair_weights.pth"
    FIGS_OUT = "hair-ai-trichologist/reports/figures"
    
    # Running for a lightweight 3 epochs since the base network features are frozen.
    # This runs comfortably inside Google Colab free tier limits in just a few minutes!
    train_hair_classifier(data_dir=RAW_DIR, output_model_path=MODEL_OUT, figures_dir=FIGS_OUT, epochs=3)
