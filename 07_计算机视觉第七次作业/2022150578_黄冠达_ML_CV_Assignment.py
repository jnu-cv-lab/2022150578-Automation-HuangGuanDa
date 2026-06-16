# 学号：2022150578
# 姓名：黄冠达
# 第8课作业：传统机器学习方法用于手写数字图像分类

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# ===================== 任务1：数据准备 =====================
print("=== 任务1：数据准备 ===")
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

print(f"图像总数: {len(images)}")
print(f"每张图像大小: {images.shape[1:]}")
print(f"类别标签: {np.unique(y)}")

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
plt.close()
print("样本图像已保存为 sample_images.png")

# ===================== 任务2：数据划分 =====================
print("\n=== 任务2：数据划分 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# ===================== 任务3：特征表示 =====================
print("\n=== 任务3：特征表示 ===")
print("将8×8的二维图像展平为64维一维特征向量")
print("传统机器学习模型只能处理一维特征输入")

# ===================== 任务4：模型训练与评估 =====================
print("\n=== 任务4：模型训练与评估 ===")
models = {
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}
y_pred_dict = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    y_pred_dict[name] = y_pred
    print(f"{name}: 准确率 = {acc:.4f}")

# ===================== 任务5：结果比较 =====================
print("\n=== 任务5：模型准确率对比 ===")
df = pd.DataFrame(list(results.items()), columns=['模型', '测试准确率'])
print(df.round(4))

best_model = max(results, key=results.get)
worst_model = min(results, key=results.get)
print(f"\n表现最好模型: {best_model}")
print(f"表现最差模型: {worst_model}")

# ===================== 任务6：错误样本分析 =====================
print("\n=== 任务6：混淆矩阵与错误样本可视化 ===")
best_pred = y_pred_dict[best_model]
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(10,8))
ConfusionMatrixDisplay(cm, display_labels=digits.target_names).plot(cmap='Blues')
plt.title(f'{best_model} 混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 错误样本展示
errors = np.where(best_pred != y_test)[0]
print(f"错误分类样本数量: {len(errors)}")

plt.figure(figsize=(12,4))
for i, idx in enumerate(errors[:8]):
    plt.subplot(1,8,i+1)
    plt.imshow(X_test[idx].reshape(8,8), cmap='gray')
    plt.title(f'T:{y_test[idx]}\nP:{best_pred[idx]}')
    plt.axis('off')
plt.savefig('error_samples.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 2022150578_黄冠达 第8课作业全部完成！")