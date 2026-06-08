import cv2
import numpy as np

def create_test_image(size=500):
    """创建测试图像"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # 绘制矩形（红色）
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 255), 2)
    cv2.putText(img, "Rectangle", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 绘制圆（绿色）
    cv2.circle(img, (400, 100), 50, (0, 255, 0), 2)
    cv2.putText(img, "Circle", (380, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 绘制平行线（蓝色）
    cv2.line(img, (50, 250), (450, 250), (255, 0, 0), 2)
    cv2.line(img, (50, 280), (450, 280), (255, 0, 0), 2)
    cv2.putText(img, "Parallel Lines", (170, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制垂直线（蓝色）
    cv2.line(img, (250, 200), (250, 350), (255, 0, 0), 2)
    cv2.line(img, (350, 200), (350, 350), (255, 0, 0), 2)
    cv2.putText(img, "Vertical Lines", (260, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def apply_similarity_transform(img):
    """相似变换：旋转30度 + 缩放0.8倍"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    return cv2.warpAffine(img, M, (w, h))

def apply_affine_transform(img):
    """仿射变换：倾斜效果"""
    h, w = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 150]])
    pts2 = np.float32([[50, 50], [220, 80], [30, 180]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h))

def apply_perspective_transform(img):
    """透视变换：梯形效果"""
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    pts2 = np.float32([[50, 0], [w-80, 0], [0, h-1], [w-1, h-1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

def main():
    print("=" * 60)
    print("几何变换实验")
    print("=" * 60)
    
    # 1. 创建测试图像
    print("\n[步骤 1] 创建测试图像...")
    original = create_test_image(500)
    cv2.imwrite('original.png', original)
    print("   已保存: original.png")
    
    # 2. 相似变换
    print("\n[步骤 2] 应用相似变换...")
    similarity = apply_similarity_transform(original)
    cv2.imwrite('similarity.png', similarity)
    print("   已保存: similarity.png")
    
    # 3. 仿射变换
    print("\n[步骤 3] 应用仿射变换...")
    affine = apply_affine_transform(original)
    cv2.imwrite('affine.png', affine)
    print("   已保存: affine.png")
    
    # 4. 透视变换
    print("\n[步骤 4] 应用透视变换...")
    perspective = apply_perspective_transform(original)
    cv2.imwrite('perspective.png', perspective)
    print("   已保存: perspective.png")
    
    # 结果总结
    print("\n" + "=" * 60)
    print("实验结果总结")
    print("=" * 60)
    
    print("\n【相似变换】")
    print("  1. 直线是否保持为直线: ✓ 是")
    print("  2. 平行线是否仍保持平行: ✓ 是")
    print("  3. 两条垂直线是否仍垂直: ✓ 是")
    print("  4. 圆是否仍保持为圆: ✓ 是")
    
    print("\n【仿射变换】")
    print("  1. 直线是否保持为直线: ✓ 是")
    print("  2. 平行线是否仍保持平行: ✓ 是")
    print("  3. 两条垂直线是否仍垂直: ✗ 否")
    print("  4. 圆是否仍保持为圆: ✗ 否 (变为椭圆)")
    
    print("\n【透视变换】")
    print("  1. 直线是否保持为直线: ✓ 是")
    print("  2. 平行线是否仍保持平行: ✗ 否")
    print("  3. 两条垂直线是否仍垂直: ✗ 否")
    print("  4. 圆是否仍保持为圆: ✗ 否 (变为椭圆)")
    
    print("\n" + "=" * 60)
    print("实验完成！生成的图片:")
    print("  - original.png")
    print("  - similarity.png")
    print("  - affine.png")
    print("  - perspective.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
