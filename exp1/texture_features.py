import numpy as np

def extract_lbp_features(image):
    """使用LBP方法提取纹理特征"""
    height, width = image.shape
    lbp_image = np.zeros_like(image)
    
    # 计算LBP图像
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = image[i, j]
            binary_pattern = 0
            
            # 8邻域像素比较
            neighbors = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                image[i, j-1],                 image[i, j+1],
                image[i+1, j-1], image[i+1, j], image[i+1, j+1]
            ]
            
            for k, neighbor in enumerate(neighbors):
                if neighbor >= center:
                    binary_pattern |= (1 << k)
            
            lbp_image[i, j] = binary_pattern
    
    return lbp_image

def calculate_lbp_histogram(lbp_image, bins=256):
    """计算LBP直方图"""
    histogram = np.zeros(bins)
    
    for pixel_value in lbp_image.flatten():
        if 0 <= pixel_value < bins:
            histogram[int(pixel_value)] += 1
    
    # 归一化
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    return histogram

def get_texture_features(image):
    """获取纹理特征"""
    lbp_image = extract_lbp_features(image)
    return calculate_lbp_histogram(lbp_image)