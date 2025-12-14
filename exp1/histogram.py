import numpy as np
from config import Config

def calculate_histogram(image_array, bins=Config.HISTOGRAM_BINS):
    """手动计算直方图"""
    histogram = np.zeros(bins)
    
    # 统计每个灰度级的像素数量
    for pixel_value in image_array.flatten():
        if 0 <= pixel_value < bins:
            histogram[pixel_value] += 1
    
    # 归一化
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    return histogram

def get_image_histogram(image_path):
    """获取图像的直方图"""
    from PIL import Image
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    img_array = np.array(img)
    
    return calculate_histogram(img_array)