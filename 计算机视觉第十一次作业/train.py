# train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 核心安全机制：解决 Linux/WSL 环境下 matplotlib 绘图因为没有图形前端而报错的问题
import matplotlib
matplotlib.use('Agg') 

# 硬件设备检测（优先使用 GPU 加速）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备: {device}")

# 双层 Transformer 核心分类网络架构
class BadmintonTransformer(nn.Module):
    def __init__(self, input_dim=132, seq_len=30, num_classes=6, num_heads=4, hidden_dim=128, num_layers=2):
        super(BadmintonTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        out = self.embedding(x) + self.pos_embedding
        out = self.transformer(out)
        out = torch.mean(out, dim=1) 
        return self.fc(out)

def main():
    DATA_PATH = "./data"
    if not os.path.exists(os.path.join(DATA_PATH, "X_train.npy")):
        print("❌ 错误：未检测到特征文件，请确保先运行了 python preprocess.py！")
        return
        
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = BadmintonTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 记录 Loss 数据用于绘制收敛曲线
    loss_history = []
    
    EPOCHS = 30
    print("\n🚀 开始迭代训练 Transformer 神经网络...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")
            
    # ====== 1. 自动生成并保存训练 Loss 曲线图 ======
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.title('Transformer Training Loss Curve', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.savefig('./loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n📸 [1/2] 训练收敛曲线图已成功保存至: ./loss_curve.png")

    # 模型测试评估
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
    target_names = ["forehand_drive", "forehand_lift", "forehand_net_shot", "forehand_clear", "backhand_drive", "backhand_net_shot"]
    
    print("\n" + "="*20 + " 📊 最终模型评估报告 " + "="*20)
    print("\n【分类评价细则报告 (Classification Report)】:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))
    
    cm = confusion_matrix(all_targets, all_preds)
    print("\n【混淆矩阵数字文本】:")
    print(cm)
    
    # ====== 2. 自动绘制并保存漂亮的彩色混淆矩阵热力图 ======
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 12, "weight": "bold"}, cbar=True)
    plt.title('Badminton Stroke Classification - Confusion Matrix', fontsize=14, pad=15)
    plt.xlabel('Predicted Action Labels', fontsize=12, labelpad=10)
    plt.ylabel('True Action Labels', fontsize=12, labelpad=10)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./confusion_matrix.png', dpi=300)
    plt.close()
    print("📸 [2/2] 高清学术混淆矩阵热力图已成功保存至: ./confusion_matrix.png")
    print("="*60)
    print("🎉 完美！所有要求的定量分析结果、文本报告、科学图表均已全部生成成功！")

if __name__ == "__main__":
    main()