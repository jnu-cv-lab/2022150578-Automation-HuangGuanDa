import cv2
import numpy as np
import matplotlib.pyplot as plt

def click_points(event, x, y, flags, param):
    """鼠标回调函数：记录点击的四个点"""
    if event == cv2.EVENT_LBUTTONDOWN:
        points = param['points']
        img = param['img']
        
        if len(points) < 4:
            points.append((x, y))
            # 在点击位置画圆标记
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, str(len(points)), (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('Select 4 corners (press ENTER when done)', img)
            print(f"已选择点 {len(points)}: ({x}, {y})")

def select_corners(image_path):
    """手动选择四个角点"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None, None
    
    original = img.copy()
    points = []
    
    cv2.namedWindow('Select 4 corners (press ENTER when done)')
    param = {'points': points, 'img': img}
    cv2.setMouseCallback('Select 4 corners (press ENTER when done)', click_points, param)
    
    print("\n请按顺序点击四个角点：")
    print("  1. 左上角")
    print("  2. 右上角")
    print("  3. 右下角")
    print("  4. 左下角")
    print("选择完成后按 ENTER 键继续...")
    
    while True:
        cv2.imshow('Select 4 corners (press ENTER when done)', img)
        key = cv2.waitKey(1)
        if key == 13:  # Enter 键
            if len(points) == 4:
                break
            else:
                print(f"请选择 {4 - len(points)} 个点，当前已选 {len(points)} 个")
        elif key == 27:  # ESC 键
            cv2.destroyAllWindows()
            return None, None
    
    cv2.destroyAllWindows()
    
    # 按顺序排列点：左上、右上、右下、左下
    src_points = np.array(points, dtype=np.float32)
    
    # 计算目标图像的尺寸（基于原始图像）
    width = max(
        np.linalg.norm(src_points[1] - src_points[0]),  # 上边
        np.linalg.norm(src_points[2] - src_points[3])   # 下边
    )
    height = max(
        np.linalg.norm(src_points[3] - src_points[0]),  # 左边
        np.linalg.norm(src_points[2] - src_points[1])   # 右边
    )
    
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    return src_points, dst_points, original

def auto_detect_corners(image_path):
    """自动检测文档四个角点（如果对比度足够）"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 膨胀连接边缘
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到最大的四边形轮廓
    max_area = 0
    best_contour = None
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                best_contour = approx
    
    if best_contour is not None:
        # 获取四个角点并排序
        points = best_contour.reshape(4, 2).astype(np.float32)
        
        # 排序：左上、右上、右下、左下
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        rect[0] = points[np.argmin(s)]      # 左上（和最小）
        rect[2] = points[np.argmax(s)]      # 右下（和最大）
        rect[1] = points[np.argmin(diff)]   # 右上（差最小）
        rect[3] = points[np.argmax(diff)]   # 左下（差最大）
        
        return rect, points, img
    
    return None, None, img

def correct_perspective(image_path, src_points, dst_points):
    """执行透视校正"""
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用变换
    width = int(dst_points[2][0] + 1)
    height = int(dst_points[2][1] + 1)
    corrected = cv2.warpPerspective(image_path if isinstance(image_path, np.ndarray) else cv2.imread(image_path), 
                                     M, (width, height))
    
    return corrected

def analyze_text_quality(original, corrected):
    """分析校正后的文字质量"""
    # 转换为灰度图
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    
    # 计算清晰度（拉普拉斯方差）
    sharpness_original = cv2.Laplacian(gray_original, cv2.CV_64F).var()
    sharpness_corrected = cv2.Laplacian(gray_corrected, cv2.CV_64F).var()
    
    print("\n" + "=" * 60)
    print("文字质量分析")
    print("=" * 60)
    print(f"  原始图像清晰度: {sharpness_original:.2f}")
    print(f"  校正后清晰度: {sharpness_corrected:.2f}")
    
    if sharpness_corrected > sharpness_original:
        print("  ✓ 校正后文字更清晰")
    else:
        print("  - 校正后文字清晰度变化不大")
    
    return sharpness_original, sharpness_corrected

def main():
    print("=" * 60)
    print("透视畸变校正程序")
    print("=" * 60)
    
    # 图片路径
    image_path = "test.jpg"
    
    # 检查图片是否存在
    import os
    if not os.path.exists(image_path):
        print(f"\n错误：找不到图片 {image_path}")
        print("请确保 test.jpg 在当前目录下")
        return
    
    print(f"\n加载图片: {image_path}")
    
    # 选择校正方式
    print("\n请选择校正方式:")
    print("  1. 自动检测文档边界（推荐用于高对比度文档）")
    print("  2. 手动选择四个角点（推荐用于复杂背景）")
    
    choice = input("\n请输入选择 (1 或 2，默认 2): ").strip()
    
    if choice == "1":
        print("\n尝试自动检测文档角点...")
        src_points, _, img = auto_detect_corners(image_path)
        if src_points is not None:
            print("自动检测成功！")
            original = img
        else:
            print("自动检测失败，切换到手动模式...")
            result = select_corners(image_path)
            if result[0] is None:
                return
            src_points, dst_points, original = result
    else:
        result = select_corners(image_path)
        if result[0] is None:
            return
        src_points, dst_points, original = result
    
    # 计算目标尺寸
    if 'dst_points' not in dir():
        width = max(
            np.linalg.norm(src_points[1] - src_points[0]),
            np.linalg.norm(src_points[2] - src_points[3])
        )
        height = max(
            np.linalg.norm(src_points[3] - src_points[0]),
            np.linalg.norm(src_points[2] - src_points[1])
        )
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
    
    print(f"\n目标尺寸: {int(dst_points[2][0]+1)} x {int(dst_points[2][1]+1)}")
    
    # 执行透视校正
    print("\n正在执行透视校正...")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    width, height = int(dst_points[2][0] + 1), int(dst_points[2][1] + 1)
    corrected = cv2.warpPerspective(original, M, (width, height))
    
    # 保存结果
    cv2.imwrite('corrected.jpg', corrected)
    print("✓ 已保存校正结果: corrected.jpg")
    
    # 在原图上标记选择的角点
    marked = original.copy()
    for i, pt in enumerate(src_points):
        cv2.circle(marked, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(marked, str(i+1), (int(pt[0])+10, int(pt[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite('marked_corners.jpg', marked)
    print("✓ 已保存角点标记图: marked_corners.jpg")
    
    # 分析文字质量
    analyze_text_quality(original, corrected)
    
    # 显示结果对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    
    # 原始图像（带标记）
    axes[0].imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original with Marked Corners")
    axes[0].axis('off')
    
    # 校正后图像
    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Perspective Corrected")
    axes[1].axis('off')
    
    # 局部放大对比（中心区域）
    h, w = corrected.shape[:2]
    center_h, center_w = h//2, w//2
    crop_size = 200
    
    if h > crop_size and w > crop_size:
        crop_original = original[center_h-crop_size//2:center_h+crop_size//2,
                                 center_w-crop_size//2:center_w+crop_size//2]
        crop_corrected = corrected[center_h-crop_size//2:center_h+crop_size//2,
                                   center_w-crop_size//2:center_w+crop_size//2]
        
        axes[2].imshow(cv2.cvtColor(crop_corrected, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Corrected (Center Zoom)")
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('correction_comparison.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 60)
    print("校正完成！生成的文件:")
    print("  - marked_corners.jpg     (标记了角点的原图)")
    print("  - corrected.jpg          (透视校正后的图像)")
    print("  - correction_comparison.png (对比图)")
    print("=" * 60)
    
    print("\n【主观评价】")
    print("请查看 corrected.jpg 图片，评估：")
    print("  1. 文字是否横平竖直？")
    print("  2. 表格线条是否平直？")
    print("  3. 文字是否有明显变形？")
    print("  4. 整体是否清晰？")

if __name__ == "__main__":
    main()
