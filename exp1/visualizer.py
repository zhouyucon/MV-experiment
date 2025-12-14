import matplotlib.pyplot as plt
import os
from PIL import Image
from config import Config

def save_results(original, sobel, custom, save_dir=Config.RESULTS_DIR):
    """保存处理结果图像"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    Image.fromarray(original).save(os.path.join(save_dir, 'original.jpg'))
    Image.fromarray(sobel).save(os.path.join(save_dir, 'sobel_filtered.jpg'))
    Image.fromarray(custom).save(os.path.join(save_dir, 'custom_filtered.jpg'))

def visualize_all_results(original, sobel, custom, histogram, texture_histogram):
    """可视化所有结果"""
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Sobel滤波结果
    plt.subplot(2, 3, 2)
    plt.imshow(sobel, cmap='gray')
    plt.title('Sobel Filter Result')
    plt.axis('off')
    
    # 自定义滤波结果
    plt.subplot(2, 3, 3)
    plt.imshow(custom, cmap='gray')
    plt.title('Custom Filter Result')
    plt.axis('off')
    
    # 颜色直方图
    plt.subplot(2, 3, 4)
    plt.bar(range(len(histogram)), histogram, width=1.0)
    plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
    
    # 纹理特征直方图
    plt.subplot(2, 3, 5)
    plt.bar(range(50), texture_histogram[:50], width=1.0)
    plt.title('Texture Features (LBP)')
    plt.xlabel('LBP Pattern')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'all_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()