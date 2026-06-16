import cv2
import numpy as np

# 1. 读取图像
img1 = cv2.imread('box.png')           # 目标物体图像
img2 = cv2.imread('box_in_scene.png')  # 包含目标的场景图像

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

# 4. 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

print(f"初始匹配数量: {len(matches)}")

# 5. 提取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 6. 使用 RANSAC 计算 Homography 矩阵
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
mask = mask.ravel().astype(bool)

inlier_count = np.sum(mask)
print(f"RANSAC 内点数量: {inlier_count}")
print(f"内点比例: {inlier_count/len(matches)*100:.2f}%")

# 7. 获取 box.png 的四个角点
h, w = img1.shape[:2]
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
print(f"\nbox.png 的四个角点坐标:")
print(f"  左上: (0, 0)")
print(f"  右上: ({w}, 0)")
print(f"  右下: ({w}, {h})")
print(f"  左下: (0, {h})")

# 8. 使用 Homography 将角点投影到场景图中
transformed_corners = cv2.perspectiveTransform(corners, H)

# 将坐标转换为整数（用于绘图）
transformed_corners_int = np.int32(transformed_corners)

print(f"\n投影到场景图中的角点坐标:")
for i, pt in enumerate(transformed_corners_int.reshape(4, 2)):
    corner_names = ["左上", "右上", "右下", "左下"]
    print(f"  {corner_names[i]}: ({pt[0]}, {pt[1]})")

# 9. 在场景图中画出目标物体的边框
img_result = img2.copy()
cv2.polylines(img_result, [transformed_corners_int], True, (0, 255, 0), 3)

# 可选：在角点上画圆标记
for pt in transformed_corners_int.reshape(4, 2):
    cv2.circle(img_result, tuple(pt), 8, (0, 0, 255), -1)

# 10. 绘制内点匹配（可选，用于可视化）
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2,
    [matches[i] for i in range(len(matches)) if mask[i]][:50],  # 显示前50个内点
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 11. 在匹配结果图上画出目标定位边框
# 注意：drawMatches 返回的图像是两张图并排的，需要调整角点坐标
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
for pt in transformed_corners_int.reshape(4, 2):
    # 在场景图部分（右侧）画角点
    cv2.circle(img_matches, (w1 + pt[0], pt[1]), 8, (0, 0, 255), -1)
# 画边框
corner_pts_scene = transformed_corners_int.reshape(4, 2)
corner_pts_scene[:, 0] += w1  # 偏移到右侧
cv2.polylines(img_matches, [corner_pts_scene], True, (0, 255, 0), 3)

# 12. 保存结果
cv2.imwrite('localization_result.jpg', img_result)
cv2.imwrite('localization_with_matches.jpg', img_matches)
print("\n结果已保存:")
print("  - localization_result.jpg (目标定位结果)")
print("  - localization_with_matches.jpg (带匹配线的定位结果)")

# 13. 显示结果
try:
    cv2.imshow('Target Localization', img_result)
    cv2.imshow('With Matches and Localization', img_matches)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("\n(无法显示窗口，图片已保存到文件)")

# 14. 输出 Homography 矩阵
print(f"\n估计的 Homography 矩阵:")
print(H)