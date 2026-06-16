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

# 4. 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

total_matches = len(matches)
print(f"初始匹配数量: {total_matches}")

# 5. 提取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 6. 使用 RANSAC 计算 Homography 矩阵
# 重投影误差阈值设为 5.0
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 7. 将 mask 转换为布尔数组
mask = mask.ravel().astype(bool)

# 8. 统计内点数量
inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
inlier_count = len(inlier_matches)
outlier_count = total_matches - inlier_count
inlier_ratio = inlier_count / total_matches * 100

print("\n=== RANSAC 结果 ===")
print(f"内点数量: {inlier_count}")
print(f"外点数量: {outlier_count}")
print(f"总匹配数量: {total_matches}")
print(f"内点比例: {inlier_ratio:.2f}%")

# 9. 显示 RANSAC 后的内点匹配
img_inliers = cv2.drawMatches(
    img1, kp1, img2, kp2,
    inlier_matches,  # 只显示内点
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 10. 同时显示所有匹配（可选，用于对比）
img_all_matches = cv2.drawMatches(
    img1, kp1, img2, kp2,
    matches[:100],  # 显示前100个匹配
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 11. 保存和显示结果
cv2.imwrite('ransac_inliers.jpg', img_inliers)
cv2.imwrite('all_matches_before_ransac.jpg', img_all_matches)
print("\n结果已保存:")
print("  - ransac_inliers.jpg (RANSAC 后的内点匹配)")
print("  - all_matches_before_ransac.jpg (RANSAC 前的所有匹配)")

# 显示结果（如果支持GUI）
try:
    cv2.imshow('All Matches (Before RANSAC)', img_all_matches)
    cv2.imshow('Inliers (After RANSAC)', img_inliers)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("\n(无法显示窗口，图片已保存到文件)")

# 12. 可选：显示 Homography 矩阵
print(f"\n估计的 Homography 矩阵:")
print(H)