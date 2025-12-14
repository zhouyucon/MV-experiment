import numpy as np
from config import Config

def apply_convolution(image, kernel):
    """手动实现卷积操作"""
    height, width = image.shape
    kernel_size = len(kernel)
    pad = kernel_size // 2
    
    # 扩展图像边界
    padded_image = np.pad(image, pad, mode='edge')
    result = np.zeros_like(image, dtype=np.float32)
    
    # 卷积计算
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(region * kernel)
    
    return result

def sobel_filter(image):
    """Sobel算子滤波"""
    kernel_x = np.array(Config.SOBEL_X_KERNEL)
    kernel_y = np.array(Config.SOBEL_Y_KERNEL)
    
    # 计算x和y方向梯度
    grad_x = apply_convolution(image, kernel_x)
    grad_y = apply_convolution(image, kernel_y)
    
    # 计算梯度幅值并归一化
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = (magnitude / magnitude.max()) * 255
    
    return magnitude.astype(np.uint8)

def custom_filter(image):
    """自定义卷积核滤波"""
    kernel = np.array(Config.CUSTOM_KERNEL)
    filtered = apply_convolution(image, kernel)
    
    # 取绝对值并归一化
    filtered = np.abs(filtered)
    filtered = (filtered / filtered.max()) * 255
    
    return filtered.astype(np.uint8)