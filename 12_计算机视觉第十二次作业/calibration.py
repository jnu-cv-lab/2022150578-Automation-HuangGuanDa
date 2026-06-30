import numpy as np
import cv2
import glob
import os

# ============ 棋盘格参数 ============
DEFAULT_CHECKERBOARD = (9, 6)
SQUARE_SIZE = 25

# ============ 辅助函数 ============
def make_pattern_points(cols, rows, square_size):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ============ 读取图片 ============
image_files = sorted(glob.glob('camera_calibration/*.png')) + \
              sorted(glob.glob('camera_calibration/*.jpg')) + \
              sorted(glob.glob('camera_calibration/*.jpeg'))

if not image_files:
    print("错误: 在 camera_calibration 文件夹中没有找到图片!")
    exit()

print(f"找到 {len(image_files)} 张图片")

objpoints = []
imgpoints = []
img_shape = None
failed_images = []

ALTERNATIVE_CONFIGS = [
    (8,6), (7,5), (10,7), (9,7), 
    (8,5), (6,4), (11,8), (9,5)
]

# ============ 角点检测 ============
for i, fname in enumerate(image_files):
    img = cv2.imread(fname)
    if img is None:
        print(f"警告: 无法读取图片 {fname}")
        failed_images.append(fname)
        continue
    
    if img_shape is None:
        img_shape = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, DEFAULT_CHECKERBOARD, None)
    used_cb = DEFAULT_CHECKERBOARD if ret else None
    
    if not ret:
        for alt_cb in ALTERNATIVE_CONFIGS:
            ret, corners = cv2.findChessboardCorners(gray, alt_cb, None)
            if ret:
                used_cb = alt_cb
                print(f"✓ 图片 {i+1}: {os.path.basename(fname)} - 使用内角点 {used_cb[0]}x{used_cb[1]}")
                break
    
    if ret and used_cb is not None:
        objp = make_pattern_points(used_cb[0], used_cb[1], SQUARE_SIZE)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        img_corners = cv2.drawChessboardCorners(img.copy(), used_cb, corners2, ret)
        cv2.imwrite(f'corners_detected_{i}.jpg', img_corners)
        print(f"✓ 图片 {i+1}/{len(image_files)}: {os.path.basename(fname)} - 成功")
    else:
        print(f"✗ 图片 {i+1}/{len(image_files)}: {os.path.basename(fname)} - 失败")
        failed_images.append(fname)

print(f"\n成功: {len(objpoints)} 张, 失败: {len(failed_images)} 张")

if len(objpoints) < 5:
    print("错误: 有效图片太少 (至少5张)")
    exit()

# ============ 标定 ============
print("\n开始相机标定...")
image_size = (img_shape[1], img_shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

# ============ 输出结果 ============
print("\n" + "="*60)
print("相机标定结果")
print("="*60)
print(f"\n图像分辨率: {img_shape[1]} x {img_shape[0]}")
print("\n相机内参矩阵 K:")
print(mtx)
print("\n畸变参数 D = [k1, k2, p1, p2, k3]:")
print(dist.ravel())

total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
mean_error = total_error / len(objpoints)
print(f"\n重投影误差: {mean_error:.4f} 像素")

# 保存结果到文本
with open('calibration_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("相机标定结果\n")
    f.write("="*60 + "\n\n")
    f.write(f"棋盘格内角点: {DEFAULT_CHECKERBOARD[0]} x {DEFAULT_CHECKERBOARD[1]}\n")
    f.write(f"方格边长: {SQUARE_SIZE} mm\n")
    f.write(f"图像分辨率: {img_shape[1]} x {img_shape[0]}\n")
    f.write(f"有效标定图片: {len(objpoints)} 张\n\n")
    f.write("相机内参矩阵 K:\n")
    f.write(str(mtx) + "\n\n")
    f.write("畸变参数 D = [k1, k2, p1, p2, k3]:\n")
    f.write(str(dist.ravel()) + "\n\n")
    f.write(f"重投影误差: {mean_error:.4f} 像素\n")

print("\n标定结果已保存到 calibration_results.txt")

# ============ 去畸变与对比图 ============
print("\n生成去畸变图像和对比图...")

# 选一张成功的图片
test_image = None
for fname in image_files:
    if fname not in failed_images:
        test_image = fname
        break

if test_image:
    img = cv2.imread(test_image)
    h, w = img.shape[:2]
    
    # 去畸变
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('original_image.jpg', img)
    cv2.imwrite('undistorted_image.jpg', dst)
    
    # 优化去畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst_opt = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imwrite('undistorted_optimized.jpg', dst_opt)
    x,y,w,h = roi
    dst_crop = dst_opt[y:y+h, x:x+w]
    cv2.imwrite('undistorted_cropped.jpg', dst_crop)
    
    # ---- 并排对比图 ----
    img_resized = img.copy()
    dst_resized = dst.copy()
    if dst_resized.shape[:2] != img_resized.shape[:2]:
        dst_resized = cv2.resize(dst_resized, (img_resized.shape[1], img_resized.shape[0]))
    
    comparison = np.hstack((img_resized, dst_resized))
    cv2.putText(comparison, "Original", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(comparison, "Undistorted", (img_resized.shape[1] + 10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imwrite('comparison.jpg', comparison)
    
    print("已生成文件:")
    print("  - original_image.jpg")
    print("  - undistorted_image.jpg")
    print("  - undistorted_optimized.jpg")
    print("  - undistorted_cropped.jpg")
    print("  - comparison.jpg (并排对比图)")
else:
    print("没有可用于去畸变的图片")

print("\n标定完成！")