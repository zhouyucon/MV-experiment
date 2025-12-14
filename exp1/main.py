import os
import numpy as np
from image_loader import load_grayscale_image, create_example_image
from filters import sobel_filter, custom_filter
from histogram import get_image_histogram
from texture_features import get_texture_features
from visualizer import save_results, visualize_all_results
from config import Config

def process_single_image(image_path):
    """处理单张图像"""
    print(f"处理图像: {image_path}")
    
    # 1. 加载图像
    original = load_grayscale_image(image_path)
    print(f"图像尺寸: {original.shape}")
    
    # 2. Sobel滤波
    sobel_result = sobel_filter(original)
    print("Sobel滤波完成")
    
    # 3. 自定义滤波
    custom_result = custom_filter(original)
    print("自定义滤波完成")
    
    # 4. 颜色直方图
    histogram = get_image_histogram(image_path)
    print("颜色直方图计算完成")
    
    # 5. 纹理特征
    texture_features = get_texture_features(original)
    print("纹理特征提取完成")
    
    # 6. 保存结果
    save_results(original, sobel_result, custom_result)
    np.save(os.path.join(Config.RESULTS_DIR, 'texture_features.npy'), texture_features)
    print("结果文件已保存")
    
    # 7. 可视化
    visualize_all_results(original, sobel_result, custom_result, histogram, texture_features)
    
    return {
        'original': original,
        'sobel': sobel_result,
        'custom': custom_result,
        'histogram': histogram,
        'texture': texture_features
    }

def main():
    """主函数"""
    print("=== 实验一：图像滤波与特征提取 ===\n")
    
    # 获取图像路径
    image_path = "test.png"
    
    if not image_path or not os.path.exists(image_path):
        print("创建示例图像...")
        example_image = create_example_image()
        from PIL import Image
        image_path = './example_image.jpg'
        Image.fromarray(example_image).save(image_path)
    
    # 处理图像
    results = process_single_image(image_path)
    
    print("\n=== 实验完成 ===")
    print(f"结果保存在: {Config.RESULTS_DIR}")

if __name__ == "__main__":
    main()