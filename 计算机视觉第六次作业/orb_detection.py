import cv2
import numpy as np

# 1. 读取图像
img1 = cv2.imread('box.png')
img2 = cv2.imread('box_in_scene.png')

# 检查图像是否成功读取
if img1 is None:
    print("错误：无法读取 box.png")
    exit()
if img2 is None:
    print("错误：无法读取 box_in_scene.png")
    exit()

# 2. 创建 ORB 检测器，设置 nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点和描述子
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# 4. 输出关键点数量
print(f"box.png 中的关键点数量: {len(keypoints1)}")
print(f"box_in_scene.png 中的关键点数量: {len(keypoints2)}")

# 5. 输出描述子维度
if descriptors1 is not None:
    print(f"描述子维度: {descriptors1.shape[1]}")
else:
    print("未检测到描述子")

# 6. 可视化关键点
img1_keypoints = cv2.drawKeypoints(
    img1, keypoints1, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

img2_keypoints = cv2.drawKeypoints(
    img2, keypoints2, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 7. 显示结果
cv2.imshow('box.png - Keypoints', img1_keypoints)
cv2.imshow('box_in_scene.png - Keypoints', img2_keypoints)

# 保存结果（可选）
cv2.imwrite('box_keypoints.jpg', img1_keypoints)
cv2.imwrite('box_in_scene_keypoints.jpg', img2_keypoints)
print("关键点可视化图像已保存")

# 等待按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()