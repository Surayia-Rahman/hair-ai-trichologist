import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_data_transformers():
    """
    Defines image transformations. Standardizes image sizing for EfficientNet 
    and applies mild data augmentation to prevent overfitting.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform

def build_model(num_classes):
    """
    Loads a pre-trained EfficientNet-B0, freezes its base feature extractor layers,
    and updates the final classification head to match our hair category counts.
    """
    # Load weights using modern Torchvision API
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Freeze feature layers so we don't destroy pre-trained weights during short Colab training
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def run_dry_test(image_dir):
    """
    Performs a rapid validation check to ensure PyTorch can load the classes 
    and construct the network architecture without crashing.
    """
    print("--- Vision Module: Running Architecture Test ---")
    if not os.path.exists(image_dir):
        print(f"[ERROR] Cannot run dry test. Path missing: {image_dir}")
        return
        
    transform = get_data_transformers()
    try:
        dataset = datasets.ImageFolder(root=image_dir, transform=transform)
        print(f"[SUCCESS] Dataset successfully loaded by PyTorch ImageFolder wrapper.")
        print(f"Detected Classes: {dataset.classes}")
        
        # Instantiate model
        model = build_model(len(dataset.classes))
        print(f"[SUCCESS] EfficientNet-B0 loaded. Classification head configured for {len(dataset.classes)} target outputs.")
        
        # Check for GPU accessibility
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Active Training Hardware Target: {device.type.upper()}")
        
    except Exception as e:
        print(f"[ERROR] Failed dry check: {str(e)}")

if __name__ == "__main__":
    RAW_DIR = "hair-ai-trichologist/data/raw/hair_images"
    run_dry_test(RAW_DIR)
