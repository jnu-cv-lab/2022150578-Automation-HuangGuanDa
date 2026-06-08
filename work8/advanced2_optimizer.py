import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

print("="*60)
print("进阶任务2：比较不同优化器")
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

# 定义CNN模型（与基础任务相同）
class SimpleCNN(nn.Module):
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

# 训练函数
def train_with_optimizer(optimizer_name, optimizer, lr, epochs=5):
    print(f"\n{'='*50}")
    print(f"优化器: {optimizer_name}, 学习率: {lr}")
    print(f"{'='*50}")
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 根据参数创建优化器
    if optimizer == 'SGD':
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == 'Adam':
        opt = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'RMSprop':
        opt = optim.RMSprop(model.parameters(), lr=lr)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
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
        'name': f"{optimizer_name} (lr={lr})",
        'test_acc': test_acc,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

# ========== 运行不同优化器比较 ==========
results = []

# 1. SGD 不同学习率
for lr in [0.01, 0.001]:
    results.append(train_with_optimizer('SGD', 'SGD', lr))

# 2. Adam 不同学习率
for lr in [0.01, 0.001]:
    results.append(train_with_optimizer('Adam', 'Adam', lr))

# 3. RMSprop
results.append(train_with_optimizer('RMSprop', 'RMSprop', 0.001))

# ========== 打印对比表格 ==========
print("\n" + "="*60)
print("优化器性能对比总结")
print("="*60)
print(f"{'优化器 (学习率)':<25} {'测试准确率':<12}")
print("-"*40)
for r in results:
    print(f"{r['name']:<25} {r['test_acc']:<12.2f}%")
print("="*60)

# ========== 绘制对比曲线 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy 对比
for r in results:
    axes[0].plot(range(1, 6), r['val_accs'], 'o-', label=r['name'])
axes[0].set_title('不同优化器验证 Accuracy 对比')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend(loc='lower right', fontsize=8)
axes[0].grid(True)

# Loss 对比
for r in results:
    axes[1].plot(range(1, 6), r['val_losses'], 'o-', label=r['name'])
axes[1].set_title('不同优化器验证 Loss 对比')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('advanced2_comparison.png', dpi=150)
print("\n已保存对比曲线图: advanced2_comparison.png")

# 打印分析结论
print("\n" + "="*60)
print("分析结论（用于实验报告）")
print("="*60)
print("""
1. Adam 通常比 SGD 收敛更快，准确率更高
2. 学习率影响：lr=0.001 通常比 lr=0.01 更稳定
3. SGD 需要 momentum 辅助才能达到较好效果
4. 对于 MNIST 这类简单任务，各优化器差异不大
""")

print("\n进阶任务2完成！")
