import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import ImprovedCNN, DATA_DIR, IMAGE_SIZE, NUM_CLASSES # Import existing components

# Constants
RESULT_DIR = 'results/experiments'
MODEL_DIR = 'models/experiments'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(use_augmentation=True, batch_size=32):
    if use_augmentation:
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
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def run_training(epochs, lr, use_aug, exp_name):
    print(f"\n>>> Running Experiment: {exp_name} (Epochs={epochs}, LR={lr}, Aug={use_aug})")
    train_loader, val_loader = get_loaders(use_augmentation=use_aug)
    model = ImprovedCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        # 매 10 에포크마다 혹은 마지막 에포크에 진행 상황 출력 (너무 많으면 지저분하므로)
        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"  [Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"  >> Result: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{exp_name}.pth"))
    
    return {"Experiment": exp_name, "Epochs": epochs, "LR": lr, "Augmentation": use_aug, 
            "Accuracy": acc, "Precision": p, "Recall": r, "F1-Score": f1}

def plot_results(df, category, x_axis, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x_axis, y="Accuracy")
    plt.title(title)
    plt.ylim(0, 1.0)
    for i, v in enumerate(df["Accuracy"]):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

def run_milestone_training(max_epochs, milestones, lr, use_aug, exp_base_name):
    print(f"\n>>> Running Milestone Experiment: {exp_base_name} (Max Epochs={max_epochs}, LR={lr}, Aug={use_aug})")
    train_loader, val_loader = get_loaders(use_augmentation=use_aug)
    model = ImprovedCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    milestone_results = []
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        current_epoch = epoch + 1
        
        # 진행 상황 출력
        if current_epoch % 5 == 0 or current_epoch == 1 or current_epoch in milestones:
            print(f"  [Epoch {current_epoch}/{max_epochs}] Loss: {epoch_loss:.4f}")

        # 마일스톤 도달 시 평가
        if current_epoch in milestones:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
            
            print(f"  [Milestone {current_epoch}] Result: Accuracy={acc:.4f}, F1={f1:.4f}")
            
            # Save intermediate model
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{exp_base_name}_epoch_{current_epoch}.pth"))
            
            milestone_results.append({
                "Experiment": f"epoch_{current_epoch}", "Epochs": current_epoch, 
                "LR": lr, "Augmentation": use_aug, 
                "Accuracy": acc, "Precision": p, "Recall": r, "F1-Score": f1
            })
            
    return milestone_results

if __name__ == "__main__":
    results = []

    # 1. Learning Rate Comparison
    print("\n[실험 1/3] Learning Rate 비교 시작 (극저지점 확인 포함)...")
    # 기존 3개 + 신규 2개 추가 (총 5단계 비교)
    lr_list = [0.00005, 0.00001] 
    for l in lr_list:
        # 극저 학습률(0.00001)이 학습될 시간을 주기 위해 30 에포크 유지
        res = run_training(epochs=30, lr=l, use_aug=True, exp_name=f"lr_{l}")
        results.append(res)
    df_lr = pd.DataFrame([r for r in results if "lr_" in r["Experiment"]])
    plot_results(df_lr, "Learning Rate", "LR", "Performance by Learning Rate (5 levels)", "exp_lr.png")

    # 3. Epoch Comparison (마일스톤 기법 적용 - 한 번의 실행으로 10, 30, 50, 100 데이터 확보)
    print("\n[실험 3/3] Epoch 마일스톤 비교 시작 (1~100 에포크 연속 진행)...")
    milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    epoch_results = run_milestone_training(max_epochs=100, milestones=milestones, lr=0.0005, use_aug=True, exp_base_name="epoch_compare")
    results.extend(epoch_results)
    
    df_epoch = pd.DataFrame(epoch_results)
    plot_results(df_epoch, "Epoch", "Epochs", "Performance Progression by Epochs", "exp_epochs.png")

    # Final Summary
    final_df = pd.DataFrame(results)
    final_df.to_csv(os.path.join(RESULT_DIR, "all_experiments.csv"), index=False)
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(final_df.to_string(index=False))
    
    best_idx = final_df["Accuracy"].idxmax()
    best_res = final_df.iloc[best_idx]
    
    print("\n" + "*"*50)
    print("BEST CONFIGURATION")
    print(f"Name: {best_res['Experiment']}")
    print(f"Accuracy: {best_res['Accuracy']:.4f}")
    print(f"Settings: Epochs={best_res['Epochs']}, LR={best_res['LR']}, Aug={best_res['Augmentation']}")
    print("*"*50)
    print(f"All graphs and CSV saved in: {RESULT_DIR}")
