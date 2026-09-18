import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration & Constants
DATA_DIR = 'data'
MODEL_DIR = 'models'
RESULT_DIR = 'results'
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 6
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

RECYCLING_GUIDE = {
    'cardboard': "종이박스: 테이프와 송장을 제거하고 납작하게 접어서 배출하세요.",
    'glass': "유리: 내용물을 비우고 라벨을 제거한 뒤 배출하세요. 깨진 유리는 종이에 싸서 쓰레기봉투에 버리세요.",
    'metal': "캔/금속: 내용물을 비우고 겉면의 플라스틱 캡 등을 제거한 뒤 배출하세요.",
    'paper': "종이: 물기에 젖지 않게 하고 묶어서 배출하세요. 비닐 코팅된 종이는 일반쓰레기입니다.",
    'plastic': "플라스틱: 내용물을 비우고 라벨을 제거한 후 압착하여 배출하세요.",
    'trash': "일반쓰레기: 재활용이 불가능한 품목입니다. 규격 봉투에 담아 배출하세요."
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 2. Data Preprocessing & Augmentation
def get_dataloaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 데이터 분할 시 시드를 고정하여 학습/검증 데이터가 섞이지 않도록 합니다.
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 3. Model Definition
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16), 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 4. Training & Evaluation Functions
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc, _, _, _ = evaluate_model(model, val_loader, device)
        
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_cnn_model.pth'))
            
    plot_training_history(history)
    return model

def evaluate_model(model, loader, device, return_all=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_all:
                all_images.append(inputs.cpu())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    if return_all:
        return acc, precision, recall, f1, np.array(all_labels), np.array(all_preds), torch.cat(all_images)
    return acc, precision, recall, f1

# 5. Visualization & Analysis Functions
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'training_history.png'))
    print(f"Training history plot saved to {RESULT_DIR}")

def analyze_results(model, loader, device):
    acc, precision, recall, f1, y_true, y_pred, images = evaluate_model(model, loader, device, return_all=True)
    
    print(f"\n--- 최종 성능 지표 ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
    print(f"Confusion Matrix saved to {RESULT_DIR}")
    
    # Misclassification Analysis
    misclassified_indices = np.where(y_true != y_pred)[0]
    print(f"\n--- 오분류 사례 분석 (총 {len(misclassified_indices)}건 중 상위 5건) ---")
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(misclassified_indices[:5]):
        true_label = CLASS_NAMES[y_true[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]) # Unnormalize
        img = np.clip(img, 0, 1)
        
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
        print(f"사례 {i+1}: 실제 {true_label} -> 예측 {pred_label}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'misclassified_examples.png'))
    print(f"Misclassified examples plot saved to {RESULT_DIR}")

def predict(image_path, model, device):
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        class_name = CLASS_NAMES[preds.item()]
        
    print(f"\n--- 예측 결과 ---")
    print(f"분류 품목: {class_name}")
    print(f"분리배출 가이드: {RECYCLING_GUIDE.get(class_name, '정보가 없습니다.')}")

# 6. Main Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EcoSortNet Evaluation & Analysis')
    parser.add_argument('--mode', type=str, choices=['train', 'analyze', 'infer'], default='analyze', help='Run mode')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--image', type=str, help='Path to image for inference')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImprovedCNN(NUM_CLASSES)
    model_path = os.path.join(MODEL_DIR, 'best_cnn_model.pth')
    
    if args.mode == 'train':
        train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
        model.to(device) # Move model to device before training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        print("Starting training...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)
        
    elif args.mode == 'analyze':
        if os.path.exists(model_path):
            model.to(device) # Move model to device before loading weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from {model_path}")
            _, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
            analyze_results(model, val_loader, device)
        else:
            print(f"Error: Model weights not found at {model_path}. Please train the model first.")
            
    elif args.mode == 'infer':
        if not args.image:
            print("Please provide an image path using --image")
        elif os.path.exists(model_path):
            model.to(device) # Move model to device
            model.load_state_dict(torch.load(model_path, map_location=device))
            predict(args.image, model, device)
        else:
            print("Error: Trained model weights not found.")
