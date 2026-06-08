import cv2
import numpy as np
import time

def orb_matching(img1, img2, nfeatures=1000):
    """ORB 特征匹配"""
    # 检测关键点和描述子
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 提取坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return 0, 0, 0, False
    
    mask = mask.ravel().astype(bool)
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(matches) * 100
    
    # 验证定位是否成功
    h, w = img1.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)
    transformed_int = np.int32(transformed)
    
    h2, w2 = img2.shape[:2]
    success = True
    for pt in transformed_int.reshape(4, 2):
        if pt[0] < 0 or pt[0] > w2 or pt[1] < 0 or pt[1] > h2:
            success = False
            break
    
    return len(matches), inliers, inlier_ratio, success


def sift_matching(img1, img2, ratio_thresh=0.75):
    """SIFT 特征匹配"""
    # 检测关键点和描述子
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # KNN 匹配
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return 0, 0, 0, False
    
    # 提取坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return 0, 0, 0, False
    
    mask = mask.ravel().astype(bool)
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(good_matches) * 100
    
    # 验证定位是否成功
    h, w = img1.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)
    transformed_int = np.int32(transformed)
    
    h2, w2 = img2.shape[:2]
    success = True
    for pt in transformed_int.reshape(4, 2):
        if pt[0] < 0 or pt[0] > w2 or pt[1] < 0 or pt[1] > h2:
            success = False
            break
    
    return len(good_matches), inliers, inlier_ratio, success


def draw_result_with_boxes(img1, img2, method_name, use_orb=True, nfeatures=1000):
    """绘制带定位边框的结果图"""
    if use_orb:
        orb = cv2.ORB_create(nfeatures=nfeatures)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is not None:
        h, w = img1.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        transformed_int = np.int32(transformed)
        
        result = img2.copy()
        cv2.polylines(result, [transformed_int], True, (0, 255, 0), 3)
        for pt in transformed_int.reshape(4, 2):
            cv2.circle(result, tuple(pt), 8, (0, 0, 255), -1)
        
        cv2.imwrite(f'{method_name}_localization.jpg', result)
        return result
    return None


def main():
    # 读取图像
    img1 = cv2.imread('box.png')
    img2 = cv2.imread('box_in_scene.png')
    
    if img1 is None or img2 is None:
        print("错误：无法读取图像文件")
        return
    
    print("="*80)
    print("ORB vs SIFT 特征匹配对比实验")
    print("="*80)
    
    # 测试 ORB
    print("\n🔵 正在运行 ORB...")
    start = time.time()
    orb_matches, orb_inliers, orb_ratio, orb_success = orb_matching(img1, img2)
    orb_time = time.time() - start
    
    # 测试 SIFT
    print("🟢 正在运行 SIFT...")
    start = time.time()
    sift_matches, sift_inliers, sift_ratio, sift_success = sift_matching(img1, img2)
    sift_time = time.time() - start
    
    # 输出表格
    print("\n" + "="*80)
    print("实验结果汇总表")
    print("="*80)
    print(f"{'方法':<12} {'匹配数量':<12} {'RANSAC内点数':<15} {'内点比例':<12} {'是否成功定位':<15} {'运行速度主观评价':<15}")
    print("-"*80)
    
    orb_speed = "快" if orb_time < 0.5 else "中等"
    sift_speed = "慢" if sift_time > 0.5 else "中等"
    
    print(f"{'ORB':<12} {orb_matches:<12} {orb_inliers:<15} {orb_ratio:.2f}%{'':<6} {'✅ 是' if orb_success else '❌ 否':<15} {orb_speed:<15} (约{orb_time:.3f}s)")
    print(f"{'SIFT':<12} {sift_matches:<12} {sift_inliers:<15} {sift_ratio:.2f}%{'':<6} {'✅ 是' if sift_success else '❌ 否':<15} {sift_speed:<15} (约{sift_time:.3f}s)")
    
    print("="*80)
    
    # 输出对比分析
    print("\n📊 对比分析:")
    print("-"*50)
    print(f"   ORB 匹配数量: {orb_matches}, 内点比例: {orb_ratio:.2f}%")
    print(f"   SIFT 匹配数量: {sift_matches}, 内点比例: {sift_ratio:.2f}%")
    
    if sift_ratio > orb_ratio:
        print(f"   ✅ SIFT 内点比例更高 ({sift_ratio:.2f}% > {orb_ratio:.2f}%)，匹配更精确")
    else:
        print(f"   ✅ ORB 内点比例更高 ({orb_ratio:.2f}% > {sift_ratio:.2f}%)，匹配更精确")
    
    if orb_time < sift_time:
        print(f"   ✅ ORB 运行更快 ({orb_time:.3f}s < {sift_time:.3f}s)，更适合实时应用")
    else:
        print(f"   ✅ SIFT 运行更快 ({sift_time:.3f}s < {orb_time:.3f}s)")
    
    # 生成可视化结果
    print("\n🎨 正在生成可视化结果...")
    draw_result_with_boxes(img1, img2, "ORB", use_orb=True)
    draw_result_with_boxes(img1, img2, "SIFT", use_orb=False)
    print("   已保存: ORB_localization.jpg, SIFT_localization.jpg")
    
    # 最终建议
    print("\n💡 最终建议:")
    print("-"*50)
    print("   • 实时性要求高 (SLAM/视频处理): 推荐 ORB")
    print("   • 精度要求高 (图像拼接/目标识别): 推荐 SIFT")
    print("   • 两者都需要配合 RANSAC 剔除错误匹配")
    print("="*80)


if __name__ == "__main__":
    main()