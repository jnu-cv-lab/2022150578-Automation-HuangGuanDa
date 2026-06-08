 实验一：Python 视觉开发环境搭建与图像基本读写

 实验信息
- 实验课程：计算机视觉实验
- 实验名称：环境搭建与图像基本读写
- 实验日期：2026年3月27日
 一、实验目的
1. 搭建 Python 视觉开发环境
2. 掌握 OpenCV 的基本图像操作（读取、显示、转换、保存）
3. 学习输出图像的基本信息（尺寸、通道数、数据类型）
4. 使用 NumPy 进行简单的像素访问和图像裁剪操作
 
 二、实验环境
- 操作系统：Windows Linux
- 编程语言：Python 
- 主要库：
  - OpenCV (`opencv-python`)
  - NumPy
 环境配置
```bash
pip install opencv-python numpy

完整代码
python
import cv2
import numpy as np

img = cv2.imread('test.jpg')

if img is None:
    print("错误：找不到图片文件！")
else:
    # 任务2：输出图像基本信息
    height, width, channels = img.shape
    print("图像尺寸:", height, "x", width)
    print("通道数:", channels)
    print("数据类型:", img.dtype)
    
    # 任务3：显示原图
    cv2.imshow('原图', img)
    
    # 任务4：转换为灰度图并显示
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('灰度图', gray)
    
    # 任务5：保存灰度图
    cv2.imwrite('gray_test.jpg', gray)
    print("灰度图已保存")
    
    # 任务6：NumPy操作 - 裁剪左上角区域并保存
    cropped = img[0:100, 0:100]
    cv2.imwrite('cropped.jpg', cropped)
    print("裁剪区域已保存")
    
    # 任务6：NumPy操作 - 输出左上角像素值
    print("左上角像素:", img[0, 0])
    
    # 等待按键后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 功能说明
任务	功能描述	                                对应代码
任务1	读取测试图片	                            cv2.imread('test.jpg')
任务2	输出图像尺寸、通道数、数据类型            	img.shape, img.dtype
任务3	显示原图	                                cv2.imshow('原图', img)
任务4	转换为灰度图并显示	                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
任务5	保存灰度图	                              cv2.imwrite('gray_test.jpg', gray)
任务6	裁剪左上角区域并保存	                      img[0:100, 0:100]
任务6	输出左上角像素值	                        print(img[0, 0])

四、实验结果
4.1 控制台输出
text
图像尺寸: 480 x 640
通道数: 3
数据类型: uint8
灰度图已保存
裁剪区域已保存
左上角像素: [142 108  85]
4.2 生成的文件
运行代码后，会生成以下文件：

gray_test.jpg - 转换后的灰度图像

cropped.jpg - 裁剪的左上角 100×100 像素区域

4.3 显示效果
程序会弹出两个图像窗口：

原图：显示原始彩色图片

灰度图：显示转换后的灰度图片



实验总结
本次实验成功完成了以下内容：

✅ 搭建了 Python 视觉开发环境
✅ 掌握了 OpenCV 的基本图像操作
✅ 学会了获取图像的基本信息
✅ 实现了图像的灰度转换和保存
✅ 使用 NumPy 进行了图像裁剪和像素访问
