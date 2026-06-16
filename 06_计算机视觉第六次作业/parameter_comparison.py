import cv2
import numpy as np

def evaluate_orb_params(nfeatures, img1, img2):
    """评估不同 nfeatures 参数下的 ORB 匹配效果"""
    
    print(f"\n{'='*50}")
    print(f"测试参数: nfeatures = {nfeatures}")
    print(f"{'='*50}")
    
    # 1. 创建 ORB 检测器
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # 2. 检测关键点和描述子
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    keypoints1_count = len(kp1)
    keypoints2_count = len(kp2)
    
    print(f"模板图关键点数: {keypoints1_count}")
    print(f"场景图关键点数: {keypoints2_count}")
    
    # 如果检测不到足够的关键点，直接返回
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("错误：检测到的关键点不足，无法进行匹配")
        return {
            'nfeatures': nfeatures,
            'kp1': keypoints1_count,
            'kp2': keypoints2_count,
            'matches': 0,
            'inliers': 0,
            'inlier_ratio': 0,
            'success': False
        }
    
    # 3. 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    total_matches = len(matches)
    print(f"匹配数量: {total_matches}")
    
    # 如果匹配点太少，无法计算 Homography
    if total_matches < 4:
        print("错误：匹配点不足，无法计算 Homography")
        return {
            'nfeatures': nfeatures,
            'kp1': keypoints1_count,
            'kp2': keypoints2_count,
            'matches': total_matches,
            'inliers': 0,
            'inlier_ratio': 0,
            'success': False
        }
    
    # 4. 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 5. RANSAC 计算 Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("错误：无法计算 Homography 矩阵")
        return {
            'nfeatures': nfeatures,
            'kp1': keypoints1_count,
            'kp2': keypoints2_count,
            'matches': total_matches,
            'inliers': 0,
            'inlier_ratio': 0,
            'success': False
        }
    
    mask = mask.ravel().astype(bool)
    inlier_count = np.sum(mask)
    inlier_ratio = inlier_count / total_matches * 100
    
    print(f"内点数量: {inlier_count}")
    print(f"内点比例: {inlier_ratio:.2f}%")
    
    # 6. 验证定位是否成功（通过投影角点检查）
    h, w = img1.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    transformed_corners_int = np.int32(transformed_corners)
    
    # 检查投影后的角点是否在图像范围内
    h2, w2 = img2.shape[:2]
    all_inside = True
    for pt in transformed_corners_int.reshape(4, 2):
        if pt[0] < 0 or pt[0] > w2 or pt[1] < 0 or pt[1] > h2:
            all_inside = False
            break
    
    success = all_inside and inlier_count >= 10
    print(f"是否成功定位: {'是' if success else '否'}")
    
    # 7. 保存可视化结果
    # 绘制内点匹配
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2,
        inlier_matches[:50],  # 最多显示50个内点
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # 在匹配图上画出定位边框
    h1, w1 = img1.shape[:2]
    for pt in transformed_corners_int.reshape(4, 2):
        cv2.circle(img_matches, (w1 + pt[0], pt[1]), 6, (0, 0, 255), -1)
    corner_pts_scene = transformed_corners_int.reshape(4, 2)
    corner_pts_scene[:, 0] += w1
    cv2.polylines(img_matches, [corner_pts_scene], True, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite(f'comparison_nfeatures_{nfeatures}.jpg', img_matches)
    print(f"结果已保存: comparison_nfeatures_{nfeatures}.jpg")
    
    return {
        'nfeatures': nfeatures,
        'kp1': keypoints1_count,
        'kp2': keypoints2_count,
        'matches': total_matches,
        'inliers': inlier_count,
        'inlier_ratio': inlier_ratio,
        'success': success
    }


def main():
    # 读取图像
    img1 = cv2.imread('box.png')
    img2 = cv2.imread('box_in_scene.png')
    
    if img1 is None or img2 is None:
        print("错误：无法读取图像文件")
        return
    
    # 测试不同的 nfeatures 参数
    nfeatures_list = [500, 1000, 2000]
    results = []
    
    for n in nfeatures_list:
        result = evaluate_orb_params(n, img1, img2)
        results.append(result)
    
    # 输出结果表格
    print(f"\n{'='*100}")
    print("实验结果汇总")
    print(f"{'='*100}")
    print(f"{'nfeatures':<12} {'模板图关键点':<12} {'场景图关键点':<12} {'匹配数量':<10} {'内点数量':<10} {'内点比例':<12} {'成功定位':<10}")
    print(f"{'-'*100}")
    
    for r in results:
        print(f"{r['nfeatures']:<12} {r['kp1']:<12} {r['kp2']:<12} {r['matches']:<10} {r['inliers']:<10} {r['inlier_ratio']:.2f}%{'':<6} {'是' if r['success'] else '否':<10}")
    
    print(f"{'='*100}")
    
    # 分析结论
    print("\n📊 分析结论:")
    print("-" * 50)
    
    best = max(results, key=lambda x: x['inlier_ratio'] if x['success'] else 0)
    print(f"✅ 最佳内点比例: nfeatures = {best['nfeatures']}, 内点比例 = {best['inlier_ratio']:.2f}%")
    
    for r in results:
        if not r['success'] and r['matches'] > 0:
            print(f"⚠️  nfeatures = {r['nfeatures']} 定位失败，原因：内点数量不足或投影超出图像范围")
    
    print("\n💡 推荐参数: nfeatures = 1000 或 2000")
    print("   - nfeatures=500 可能因关键点不足导致匹配质量下降")
    print("   - nfeatures=2000 能检测更多关键点，但计算时间增加")
    print("   - nfeatures=1000 在精度和效率之间取得较好平衡")


if __name__ == "__main__":
    main()