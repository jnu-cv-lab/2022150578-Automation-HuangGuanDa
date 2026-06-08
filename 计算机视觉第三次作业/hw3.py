import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像（灰度）
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("无法读取 1.jpg，请确认文件存在！")
    exit()

h, w = img.shape
print(f"原图尺寸: {w} x {h}")

# 2. 下采样（1/2）
scale = 0.5
small_size = (int(w * scale), int(h * scale))

# 直接缩小 vs 先高斯平滑再缩小
small_direct = cv2.resize(img, small_size, interpolation=cv2.INTER_AREA)
img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
small_gauss = cv2.resize(img_gauss, small_size, interpolation=cv2.INTER_AREA)

cv2.imwrite('small_direct.jpg', small_direct)
cv2.imwrite('small_gauss.jpg', small_gauss)

# 3. 恢复（上采样回原尺寸）
methods = {
    'Nearest': cv2.INTER_NEAREST,
    'Bilinear': cv2.INTER_LINEAR,
    'Bicubic': cv2.INTER_CUBIC
}

restored = {}
for name, inter in methods.items():
    restored[name + '_direct'] = cv2.resize(small_direct, (w, h), interpolation=inter)
    restored[name + '_gauss']  = cv2.resize(small_gauss,  (w, h), interpolation=inter)

# 4. 保存空间域对比图（原图 + 缩小 + 3种恢复）
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(small_gauss, cmap='gray'); plt.title('Downsampled (Gaussian)'); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(restored['Bilinear_gauss'], cmap='gray'); plt.title('Restored - Bilinear'); plt.axis('off')
plt.subplot(2, 3, 4); plt.imshow(restored['Bicubic_gauss'], cmap='gray'); plt.title('Restored - Bicubic'); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(restored['Nearest_gauss'], cmap='gray'); plt.title('Restored - Nearest'); plt.axis('off')

plt.tight_layout()
plt.savefig('spatial_comparison.png', dpi=400, bbox_inches='tight')
print("空间域对比图已保存: spatial_comparison.png")

# 5. 保存频域分析图
def show_fft(image, title):
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    plt.imshow(mag, cmap='gray')
    plt.title(title)
    plt.axis('off')

def show_dct(image, title):
    dct = cv2.dct(np.float32(image))
    plt.imshow(np.log(np.abs(dct) + 1), cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(15, 8))
plt.subplot(2, 4, 1); show_fft(img, 'Original FFT')
plt.subplot(2, 4, 2); show_fft(small_gauss, 'Downsampled FFT')
plt.subplot(2, 4, 3); show_fft(restored['Bilinear_gauss'], 'Bilinear FFT')
plt.subplot(2, 4, 4); show_fft(restored['Bicubic_gauss'], 'Bicubic FFT')
plt.subplot(2, 4, 5); show_dct(img, 'Original DCT')
plt.subplot(2, 4, 6); show_dct(restored['Bilinear_gauss'], 'Bilinear DCT')
plt.subplot(2, 4, 7); show_dct(restored['Bicubic_gauss'], 'Bicubic DCT')
plt.subplot(2, 4, 8); show_dct(restored['Nearest_gauss'], 'Nearest DCT')

plt.tight_layout()
plt.savefig('frequency_analysis.png', dpi=400, bbox_inches='tight')
print("频域分析图已保存: frequency_analysis.png")

print("\n实验完成！现在应该有 spatial_comparison.png 和 frequency_analysis.png 了。")