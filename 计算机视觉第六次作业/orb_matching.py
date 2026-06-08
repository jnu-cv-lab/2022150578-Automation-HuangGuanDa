import cv2
import numpy as np

# 1. 读取图像
img1 = cv2.imread('box.png')
img2 = cv2.imread('box_in_scene.png')

if img1 is None or img2 is None:
    print("错误：无法读取图像文件")
    exit()

# 2. 创建 ORB 检测器
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点和描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print(f"box.png 关键点数量: {len(kp1)}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}")
print(f"描述子维度: {des1.shape[1]}")

# 4. 创建暴力匹配器（使用汉明距离）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 5. 执行匹配
matches = bf.match(des1, des2)

# 6. 按照匹配距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

# 7. 输出总匹配数量
print(f"总匹配数量: {len(matches)}")

# 8. 显示前 50 个匹配结果（或全部，如果不足50个）
num_display = min(50, len(matches))
print(f"显示前 {num_display} 个匹配结果")

# 9. 绘制匹配结果
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2,
    matches[:num_display],  # 取前 num_display 个匹配
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 10. 显示和保存结果
cv2.imshow('ORB Matches (Top 50)', img_matches)
cv2.imwrite('orb_matches.jpg', img_matches)
print("匹配结果已保存为 orb_matches.jpg")

# 打印前10个匹配的距离（用于观察）
print("\n前10个匹配的距离:")
for i, m in enumerate(matches[:10]):
    print(f"  匹配 {i+1}: 距离 = {m.distance}")

cv2.waitKey(0)
cv2.destroyAllWindows()