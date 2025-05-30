import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Params
BATCH, LR, EPOCHS = 64, 1e-3, 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_data():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)

    train_size = int(0.8 * len(train_full))
    train_set, val_set = torch.utils.data.random_split(train_full, [train_size, len(train_full) - train_size])
    val_set.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=test_tf)

    loader = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=2)
    return loader(train_set, True), loader(val_set, False), loader(test_set, False)

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        def conv_block(in_c, out_c): return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x): return self.classifier(self.conv3(self.conv2(self.conv1(x))))

def train_epoch(model, loader, loss_fn, optim):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), 100 * correct / total

def run(model, loader, loss_fn=None, return_preds=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(1)
            if loss_fn:
                total_loss += loss_fn(out, y).item()
                correct += (preds == y).sum().item()
                total += y.size(0)
            if return_preds:
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
    if return_preds and not loss_fn: return y_true, y_pred
    if loss_fn and not return_preds: return total_loss / len(loader), 100 * correct / total
    if loss_fn and return_preds: return total_loss / len(loader), 100 * correct / total, y_true, y_pred

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.tight_layout(); plt.show()

def plot_learning_curves(train_loss, val_loss, train_acc, val_acc):
    sns.set_theme(style="whitegrid")
    epochs = range(1, len(train_loss)+1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(epochs, train_loss, label='Train'), ax[0].plot(epochs, val_loss, label='Val')
    ax[1].plot(epochs, train_acc, label='Train'), ax[1].plot(epochs, val_acc, label='Val')
    for a, t, y in zip(ax, ['Loss', 'Accuracy'], ['Loss', 'Accuracy (%)']):
        a.set_title(f'{t} over Epochs'); a.set_xlabel('Epoch'); a.set_ylabel(y); a.legend()
    plt.tight_layout(); plt.show()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-3):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_loss, self.early_stop = 0, float('inf'), False
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()
    model = CNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    early_stop = EarlyStopping()

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_acc = 0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = run(model, val_loader, loss_fn)
        scheduler.step()
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), 'best_model.pth')
        train_loss.append(tr_loss), val_loss.append(va_loss)
        train_acc.append(tr_acc), val_acc.append(va_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train: {tr_loss:.4f}/{tr_acc:.2f}% - Val: {va_loss:.4f}/{va_acc:.2f}%")
        early_stop(va_loss)
        if early_stop.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    y_true, y_pred = run(model, test_loader, return_preds=True)
    test_acc = 100 * np.mean(np.array(y_pred) == np.array(y_true))
    print(f"\nBest Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASSES)
    plot_learning_curves(train_loss, val_loss, train_acc, val_acc)
    print("\nClass-wise Accuracy:")
    for cls, acc in zip(CLASSES, cm.diagonal() / cm.sum(1) * 100):
        print(f"{cls:>12}: {acc:5.2f}%")
