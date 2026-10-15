import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 50
CSV_FILE = r'D:\CORE_PROJECTSSSS\VS code projects\Pyhton prjs\FER'  # Ensure this file is in the same directory
EMOTIONS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Check for NVIDIA GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 2. Custom Dataset Class
# -----------------------------------------------------------------------------
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, split='Training', transform=None):
        print(f"Loading {split} data from {csv_file}...")
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"ERROR: '{csv_file}' not found. Please download fer2013.csv.")
            exit()
            
        self.data = self.data[self.data['Usage'] == split]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx]['pixels']
        label = int(self.data.iloc[idx]['emotion'])
        
        # Convert string of pixels to numpy array
        face_img = np.fromstring(pixels, sep=' ').astype(np.uint8)
        face_img = face_img.reshape(48, 48)
        face_img = Image.fromarray(face_img)
        
        if self.transform:
            face_img = self.transform(face_img)
            
        return face_img, label

# -----------------------------------------------------------------------------
# 3. Model Architecture (ResNet18 Custom)
# -----------------------------------------------------------------------------
from torchvision.models import resnet18

def get_model():
    model = resnet18(weights=None)
    # Modify first layer for Grayscale (1 channel)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify last layer for 7 emotions
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model.to(device)

# -----------------------------------------------------------------------------
# 4. Training Helper Functions (With Progress Bars)
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    acc = 100. * correct / total
    return running_loss / len(loader), acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    loop = tqdm(loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    return running_loss / len(loader), acc

# -----------------------------------------------------------------------------
# 5. Interactive User Demo
# -----------------------------------------------------------------------------
def user_interaction_loop(model, test_dataset):
    model.eval()
    print("\n" + "="*50)
    print(" INTERACTIVE DEMO MODE")
    print("="*50)
    print("Available emotions: angry, disgust, fear, happy, sad, surprise, neutral")
    print("Type 'exit' to quit.")

    print("Scanning test set for examples...")
    indices_by_emotion = {emotion_name: [] for emotion_name in EMOTIONS.values()}
    indices = range(len(test_dataset))
    
    with torch.no_grad():
        for i in tqdm(indices, desc="Indexing Test Data"):
            img, label = test_dataset[i]
            img_tensor = img.unsqueeze(0).to(device)
            output = model(img_tensor)
            _, predicted = output.max(1)
            pred_idx = predicted.item()
            
            if pred_idx == label:
                emotion_name = EMOTIONS[label]
                indices_by_emotion[emotion_name].append(i)

    while True:
        user_input = input("\nEnter an emotion to see a face: ").strip().lower().capitalize()
        
        if user_input.lower() == 'exit':
            break
            
        found = False
        for key, val in EMOTIONS.items():
            if val == user_input:
                found = True
                possible_indices = indices_by_emotion[val]
                
                if not possible_indices:
                    print(f"Sorry, the model didn't correctly recognize any '{user_input}' faces in the test set.")
                    continue
                
                idx = np.random.choice(possible_indices)
                img_tensor, label = test_dataset[idx]
                
                # Unnormalize for display
                img_display = img_tensor.cpu().numpy().squeeze()
                img_display = (img_display * 0.5) + 0.5
                img_display = np.clip(img_display, 0, 1)
                
                plt.figure(figsize=(4, 4))
                plt.imshow(img_display, cmap='gray')
                plt.title(f"Model Prediction: {user_input}\nTrue Label: {EMOTIONS[label]}")
                plt.axis('off')
                plt.show()
                break
        
        if not found:
            print("Invalid emotion. Try: Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust")

# -----------------------------------------------------------------------------
# 6. Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FER2013 Emotion Recognition')
    parser.add_argument('--demo', action='store_true', help='Skip training and run interactive demo')
    args = parser.parse_args()

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found. Training will be slow on CPU.")

    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(48, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Always load validation set (needed for demo and eval)
    val_dataset = FER2013Dataset(CSV_FILE, split='PublicTest', transform=val_transform) 
    
    model = get_model()

    if not args.demo:
        # --- Training Mode ---
        # Only load training data if we are NOT in demo mode
        train_dataset = FER2013Dataset(CSV_FILE, split='Training', transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        best_acc = 0.0
        print(f"\nStarting training for {EPOCHS} epochs...")
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            
            scheduler.step(val_acc)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'fer2013_resnet18_best.pth')
                print("  -> Saved best model")

        print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.2f}%")
    else:
        print("\nSkipping training mode...")
    
    # --- Demo Mode ---
    if os.path.exists('fer2013_resnet18_best.pth'):
        print("Loading saved model weights...")
        model.load_state_dict(torch.load('fer2013_resnet18_best.pth', map_location=device))
        user_interaction_loop(model, val_dataset)
    else:
        print("Error: 'fer2013_resnet18_best.pth' not found. Please run without --demo first to train the model.")