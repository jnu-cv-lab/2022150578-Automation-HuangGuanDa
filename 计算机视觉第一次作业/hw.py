import cv2
import numpy as np

img = cv2.imread('test.jpg')

if img is None:
    print("错误：找不到图片文件！")
else:
    height, width, channels = img.shape
    print("图像尺寸:", height, "x", width)
    print("通道数:", channels)
    print("数据类型:", img.dtype)
    
    cv2.imshow('原图', img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('灰度图', gray)
    
    cv2.imwrite('gray_test.jpg', gray)
    print("灰度图已保存")
    
    cropped = img[0:100, 0:100]
    cv2.imwrite('cropped.jpg', cropped)
    print("裁剪区域已保存")
    
    print("左上角像素:", img[0, 0])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()