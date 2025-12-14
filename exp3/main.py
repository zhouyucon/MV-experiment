import numpy as np
import torch
from torchvision.datasets import MNIST
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import os

# 设置设备为CPU
device = torch.device('cpu')
print("使用设备: CPU")

class HandwrittenDigitRecognizer:
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.model = None
        
    def load_mnist_data(self):
        """加载MNIST数据集（完全按照图片中的方式）"""
        print("正在加载MNIST数据集...")
        
        # 按照图片中的data_tf函数
        def data_tf(x):
            x = np.array(x, dtype='float32') / 255
            x = (x - 0.5) / 0.5  # 标准化
            x = x.reshape(-1,)   # 拉平成一个一维向量（用于MLP）
            x = torch.from_numpy(x)
            return x
        
        # 下载训练集和测试集（按照图片中的方式）
        train_set = MNIST('./data', train=True, transform=data_tf, download=True)
        test_set = MNIST('./data', train=False, transform=data_tf, download=True)
        
        # 查看第一个数据（按照图片中的方式）
        a_data, a_label = train_set[0]
        print(f"数据形状: {a_data.shape}")  # 应该是(784,)
        print(f"第一个数据的标签: {a_label}")
        
        # 创建数据加载器
        train_data = DataLoader(train_set, batch_size=64, shuffle=True)
        test_data = DataLoader(test_set, batch_size=1000, shuffle=False)
        
        print(f"训练集大小: {len(train_set)}")
        print(f"测试集大小: {len(test_set)}")
        
        return train_data, test_data, train_set, test_set
    
    def create_mlp_model(self):
        """创建MLP模型（按照图片中的神经网络结构思路）"""
        class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                # 输入层 784 -> 隐藏层 -> 输出层 10
                self.model = nn.Sequential(
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                x = self.model(x)
                return x
        
        model = MLP().to(device)
        print("MLP模型创建完成")
        return model
    
    def create_cnn_model(self):
        """创建CNN模型（备选方案）"""
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 32, 5, 1, 2),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, 5, 1, 2),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.fc = nn.Sequential(
                    nn.Linear(64 * 7 * 7, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 10)
                )
            
            def forward(self, x):
                x = x.view(-1, 1, 28, 28)  # 重塑为图像格式
                x = self.conv1(x)
                x = self.conv2(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = CNN().to(device)
        print("CNN模型创建完成")
        return model
    
    def train_model(self, use_mlp=True, epochs=5):
        """训练模型"""
        train_data, test_data, train_set, test_set = self.load_mnist_data()
        
        # 选择模型
        if use_mlp:
            print("使用MLP模型训练...")
            self.model = self.create_mlp_model()
        else:
            print("使用CNN模型训练...")
            self.model = self.create_cnn_model()
        
        # 定义损失函数和优化器（按照图片中的思路）
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        
        losses = []
        accuracies = []
        
        print("开始训练...")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (data, label) in enumerate(train_data):
                # 移动到CPU设备
                data = data.to(device)
                label = label.to(device)
                
                # 前向传播
                output = self.model(data)
                loss = criterion(output, label)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            
            epoch_loss = running_loss / len(train_data)
            epoch_acc = 100 * correct / total
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # 测试模型
        test_acc = self.evaluate_model(test_data)
        print(f'测试准确率: {test_acc:.2f}%')
        
        return losses, accuracies, test_acc
    
    def evaluate_model(self, test_data):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, label in test_data:
                data = data.to(device)
                label = label.to(device)
                
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def preprocess_custom_image(self, image_path):
        """预处理自定义图像（用于学号识别）"""
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path
        
        if image is None:
            raise ValueError("无法读取图像文件")
        
        # 二值化处理
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digits = []
        bounding_boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > 10 and h > 20:  # 过滤太小的区域
                digit_region = binary_image[y:y+h, x:x+w]
                digit_resized = cv2.resize(digit_region, (28, 28))
                
                # 使用与训练数据相同的预处理
                digit_normalized = ((digit_resized / 255.0) - 0.5) / 0.5
                digit_flattened = digit_normalized.reshape(-1)  # 拉平为一维向量
                
                digits.append(digit_flattened)
                bounding_boxes.append((x, y, w, h))
        
        # 按x坐标排序
        if bounding_boxes:
            sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i][0])
            sorted_digits = [digits[i] for i in sorted_indices]
            sorted_boxes = [bounding_boxes[i] for i in sorted_indices]
        else:
            sorted_digits, sorted_boxes = [], []
        
        return sorted_digits, sorted_boxes, binary_image
    
    def predict_student_id(self, image_path):
        """预测学号"""
        if self.model is None:
            raise ValueError("请先训练模型！")
        
        digits, bounding_boxes, processed_image = self.preprocess_custom_image(image_path)
        
        if len(digits) == 0:
            return "未检测到数字", processed_image, []
        
        self.model.eval()
        predicted_digits = []
        
        with torch.no_grad():
            for digit in digits:
                digit_tensor = torch.from_numpy(digit).float().unsqueeze(0).to(device)
                output = self.model(digit_tensor)
                _, predicted = torch.max(output.data, 1)
                predicted_digits.append(predicted.item())
        
        student_id = ''.join(map(str, predicted_digits))
        return student_id, processed_image, list(zip(bounding_boxes, predicted_digits))
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有训练好的模型可以保存！")
        
        torch.save(self.model.state_dict(), filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath, use_mlp=True):
        """加载模型"""
        if use_mlp:
            self.model = self.create_mlp_model()
        else:
            self.model = self.create_cnn_model()
        
        self.model.load_state_dict(torch.load(filepath, map_location=device))
        self.model.to(device)
        print(f"模型已从 {filepath} 加载")

def create_sample_student_id():
    """创建包含学号的示例图像"""
    # 创建白色背景
    image = np.ones((150, 500), dtype=np.uint8) * 255
    
    # 学号：2022800098
    digits = [2, 0, 2, 2, 8, 0, 0, 0, 9, 8]
    positions = [(30, 80), (70, 80), (110, 80), (150, 80), (190, 80),
                (230, 80), (270, 80), (310, 80), (350, 80), (390, 80)]
    
    for (x, y), digit in zip(positions, digits):
        cv2.putText(image, str(digit), (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    
    cv2.imwrite('student_id.jpg', image)
    print("示例学号图片已创建: student_id.jpg")
    return image

def plot_results(train_losses, train_accuracies, test_image, processed_image, detection_results):
    """绘制结果"""
    plt.figure(figsize=(15, 10))
    
    # 训练历史
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies)
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    # 原始图像
    plt.subplot(2, 2, 3)
    plt.imshow(test_image, cmap='gray')
    plt.title('原始学号图像')
    plt.axis('off')
    
    # 处理结果
    plt.subplot(2, 2, 4)
    result_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h), digit in detection_results:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, str(digit), (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('数字检测结果')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""

    
    # 创建识别器
    recognizer = HandwrittenDigitRecognizer('./data')
    
    # 训练模型（使用MLP，与图片中一致）
    print("\n1. 模型训练阶段")
    train_losses, train_accuracies, test_accuracy = recognizer.train_model(
        use_mlp=True, epochs=5
    )
    
    # 保存模型
    recognizer.save_model('handwritten_digit_model.pth')
    
    # 创建测试图像
    print("\n2. 创建测试图像")
    test_image = create_sample_student_id()
    
    # 识别学号
    print("\n3. 学号识别")
    student_id, processed_image, detection_results = recognizer.predict_student_id('student_id.jpg')
    
    print(f"\n识别结果: {student_id}")
    print(f"期望学号: 2022800098")
    
    # 绘制结果
    plot_results(train_losses, train_accuracies, test_image, processed_image, detection_results)
    
    # 实验总结
    print("\n4. 实验总结")
    print("=" * 30)
    print(f"最终测试准确率: {test_accuracy:.2f}%")
    print(f"学号识别结果: {student_id}")
    print(f"使用的模型: MLP (与图片中一致)")
    print(f"运行设备: CPU")
    print(f"数据路径: ./data")

if __name__ == "__main__":
    main()

