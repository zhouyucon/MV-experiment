import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 设置路径
DATA_DIR = Path('./data/coco')
MODEL_DIR = DATA_DIR / 'models'
RESULTS_DIR = Path('./results')

class SimpleBicycleDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建目录
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # 下载模型
        self.model_path = self.download_model()
        self.model = self.load_model()
        
        # COCO自行车类别ID
        self.bicycle_id = 2
    
    def download_model(self):
        """下载COCO预训练模型"""
        model_path = MODEL_DIR / 'fasterrcnn.pth'
        
        if model_path.exists():
            print("模型已存在")
            return model_path
        
        print("下载COCO预训练模型...")
        try:
            # 使用torchvision直接下载
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            torch.save(model.state_dict(), model_path)
            print("下载完成")
            return model_path
        except:
            print("下载失败，使用随机权重")
            return None
    
    def load_model(self):
        """加载模型"""
        if self.model_path and self.model_path.exists():
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
            model.load_state_dict(torch.load(self.model_path))
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        
        model.to(self.device)
        model.eval()
        return model
    
    def load_image(self, image_path):
        """加载图像"""
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    
    def preprocess(self, image):
        """预处理图像"""
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, 
                                 mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        return image_tensor.unsqueeze(0).to(self.device)
    
    def detect(self, image_path):
        """检测自行车"""
        image = self.load_image(image_path)
        image_tensor = self.preprocess(image)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # 提取自行车检测结果
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label == self.bicycle_id and score > 0.5:
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(score)
                })
        
        return image, detections
    
    def visualize(self, image, detections, save_path=None):
        """可视化结果"""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            
            # 绘制边界框
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           fill=False, edgecolor='red', linewidth=2))
            plt.text(x1, y1-10, f'Bicycle {i+1}: {score:.2f}', 
                    color='red', fontsize=12, weight='bold')
        
        plt.title(f'Detected {len(detections)} Bicycles')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def save_results(self, detections, save_path):
        """保存结果"""
        results = {
            'detections': detections,
            'count': len(detections)
        }
        np.save(save_path, results)
        print(f"Results saved: {save_path}")

def main():
    print("实验四：共享单车检测")
    print("=" * 40)
    
    # 初始化检测器
    detector = SimpleBicycleDetector()
    
    # 查找图像文件
    image_files = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png'))
    
    if not image_files:
        print("没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    for i, img_path in enumerate(image_files):
        print(f"\n处理: {img_path.name}")
        
        # 检测
        image, detections = detector.detect(str(img_path))
        print(f"检测到 {len(detections)} 辆自行车")
        
        # 显示结果
        for j, det in enumerate(detections):
            print(f"  自行车{j+1}: 位置{det['bbox']}, 分数{det['score']:.2f}")
        
        # 保存结果
        result_img = RESULTS_DIR / f"result_{i+1}.jpg"
        detector.visualize(image, detections, str(result_img))
        
        result_data = RESULTS_DIR / f"detection_{i+1}.npy"
        detector.save_results(detections, str(result_data))
    
    print("\n检测完成！")

if __name__ == "__main__":
    main()
