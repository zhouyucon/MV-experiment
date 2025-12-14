from PIL import Image
import numpy as np

def load_grayscale_image(image_path):
    """加载图像并转换为灰度图"""
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)

def create_example_image():
    """创建示例图像"""
    width, height = 400, 300
    image = np.zeros((height, width), dtype=np.uint8)
    
    # 创建渐变背景
    for i in range(height):
        for j in range(width):
            image[i, j] = int((i + j) / (height + width) * 200 + 55)
    
    # 添加圆形
    center_x, center_y = width // 2, height // 2
    for i in range(height):
        for j in range(width):
            distance = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            if distance < 50:
                image[i, j] = 255
            elif 50 <= distance < 60:
                image[i, j] = 100
    
    return image