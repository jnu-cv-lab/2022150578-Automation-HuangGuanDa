import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("进阶任务1：修改网络结构")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("加载 MNIST 数据集...")
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_set, val_set = random_split(full_train, [50000, 10000])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# ========== 模型1：原始模型（基线） ==========
class BaselineCNN(nn.Module):
    """原始模型：2个卷积层 + 1个全连接层"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc(x)

# ========== 模型2：改进模型1（增加卷积层） ==========
class DeeperCNN(nn.Module):
    """改进1：增加第三个卷积层"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        # 经过3次池化后：28 -> 14 -> 7 -> 3
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128*3*3, 256), nn.ReLU(), nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)

# ========== 模型3：改进模型2（增加Dropout和BN） ==========
class AdvancedCNN(nn.Module):
    """改进2：增加BatchNorm + Dropout + 更多卷积核"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)

# ========== 训练函数 ==========
def train_and_evaluate(model, model_name, epochs=5):
    print(f"\n{'='*50}")
    print(f"训练模型: {model_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    
    # 测试
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100 * test_correct / test_total
    print(f"\n测试集准确率: {test_acc:.2f}%")
    
    return {
        'name': model_name,
        'params': sum(p.numel() for p in model.parameters()),
        'test_acc': test_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# ========== 运行三个模型比较 ==========
results = []

# 1. 基线模型
results.append(train_and_evaluate(BaselineCNN(), "基线模型 (2卷积层)"))

# 2. 加深模型
results.append(train_and_evaluate(DeeperCNN(), "加深模型 (3卷积层)"))

# 3. 高级模型
results.append(train_and_evaluate(AdvancedCNN(), "高级模型 (BN+Dropout)"))

# ========== 打印对比结果 ==========
print("\n" + "="*60)
print("模型性能对比总结")
print("="*60)
print(f"{'模型名称':<25} {'参数量':<12} {'测试准确率':<12}")
print("-"*50)
for r in results:
    print(f"{r['name']:<25} {r['params']:<12,} {r['test_acc']:<12.2f}%")
print("="*60)

# ========== 绘制对比曲线 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 对比
for r in results:
    axes[0].plot(range(1, 6), r['val_losses'], 'o-', label=r['name'])
axes[0].set_title('不同模型验证 Loss 对比')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy 对比
for r in results:
    axes[1].plot(range(1, 6), r['val_accs'], 'o-', label=r['name'])
axes[1].set_title('不同模型验证 Accuracy 对比')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('advanced1_comparison.png', dpi=150)
print("\n已保存对比曲线图: advanced1_comparison.png")

print("\n进阶任务1完成！")
print("观察结论:")
print("1. 加深模型（3层卷积）相比基线模型，准确率可能提升")
print("2. 加入BatchNorm和Dropout可以缓解过拟合")
print("3. 参数量增加会带来更好的表达能力，但也需要更多训练时间")
