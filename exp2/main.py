import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetection:
    def __init__(self):
        pass
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blur, 50, 150)
        
        return edges
    
    def region_of_interest(self, image):
        """定义感兴趣区域（ROI）"""
        height, width = image.shape
        # 创建一个掩码
        mask = np.zeros_like(image)
        
        # 定义多边形顶点（根据车道位置调整）
        polygon = np.array([[
            (width * 0.1, height),  # 左下角
            (width * 0.45, height * 0.6),  # 左上角
            (width * 0.55, height * 0.6),  # 右上角
            (width * 0.9, height)  # 右下角
        ]], np.int32)
        
        # 填充多边形
        cv2.fillPoly(mask, polygon, 255)
        
        # 与原始图像进行与操作
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def detect_lanes_with_hough(self, edges):
        """使用霍夫变换检测直线"""
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 
                               rho=1, 
                               theta=np.pi/180, 
                               threshold=15, 
                               minLineLength=40, 
                               maxLineGap=20)
        
        return lines
    
    def average_slope_intercept(self, lines, image_shape):
        """将检测到的直线按左右车道线分类并求平均"""
        if lines is None:
            return None, None
        
        left_lines = []  # 左车道线
        right_lines = []  # 右车道线
        left_weights = []  # 权重（根据长度）
        right_weights = []  # 权重（根据长度）
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:  # 避免除零错误
                continue
                
            # 计算斜率和截距
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            
            # 过滤掉水平线（斜率接近0）
            if abs(slope) < 0.5:
                continue
                
            # 根据斜率正负分类左右车道线
            if slope < 0:  # 左车道线（斜率负）
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:  # 右车道线（斜率正）
                right_lines.append((slope, intercept))
                right_weights.append(length)
        
        # 计算加权平均
        left_lane = None
        right_lane = None
        
        if left_lines and left_weights:
            left_avg = np.average(left_lines, axis=0, weights=left_weights)
            left_lane = self.make_line_points(left_avg, image_shape)
            
        if right_lines and right_weights:
            right_avg = np.average(right_lines, axis=0, weights=right_weights)
            right_lane = self.make_line_points(right_avg, image_shape)
            
        return left_lane, right_lane
    
    def make_line_points(self, line_params, image_shape):
        """根据斜率和截距生成直线的端点"""
        slope, intercept = line_params
        y1 = image_shape[0]  # 图像底部
        y2 = int(y1 * 0.6)  # 图像顶部附近
        
        # 确保斜率不为零
        if abs(slope) < 1e-5:
            slope = 1e-5
            
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        return ((x1, y1), (x2, y2))
    
    def draw_lanes(self, image, left_lane, right_lane):
        """在图像上绘制车道线"""
        lane_image = np.zeros_like(image)
        
        if left_lane is not None and right_lane is not None:
            # 绘制左右车道线
            cv2.line(lane_image, left_lane[0], left_lane[1], (0, 255, 0), 10)
            cv2.line(lane_image, right_lane[0], right_lane[1], (0, 255, 0), 10)
            
            # 填充车道区域
            points = np.array([left_lane[0], left_lane[1], right_lane[1], right_lane[0]])
            cv2.fillPoly(lane_image, [points], (0, 255, 0, 100))
        
        # 将车道线叠加到原图上
        result = cv2.addWeighted(image, 0.8, lane_image, 0.5, 0)
        
        return result
    
    def process_image(self, image_path):
        """处理单张图像的主函数"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 预处理
        edges = self.preprocess_image(image)
        
        # ROI提取
        roi_edges = self.region_of_interest(edges)
        
        # 霍夫变换检测直线
        lines = self.detect_lanes_with_hough(roi_edges)
        
        # 处理检测到的直线
        left_lane, right_lane = self.average_slope_intercept(lines, image.shape)
        
        # 绘制车道线
        result_image = self.draw_lanes(image, left_lane, right_lane)
        
        return result_image, edges, roi_edges
    
    def visualize_results(self, original, edges, roi_edges, result, image_path):
        """可视化处理结果"""
        plt.figure(figsize=(15, 10))
        
        # 原图
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # 边缘检测结果
        plt.subplot(2, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')
        
        # ROI区域边缘
        plt.subplot(2, 2, 3)
        plt.imshow(roi_edges, cmap='gray')
        plt.title('ROI Edges')
        plt.axis('off')
        
        # 最终结果
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Lane Detection Result')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存结果图像
        output_path = image_path.replace('.jpg', '_result.jpg')
        cv2.imwrite(output_path, result)
        print(f"结果已保存到: {output_path}")
        
        plt.show()

def main():
    """主函数"""
    # 创建车道线检测对象
    lane_detector = LaneDetection()
    
    # 图像路径（请替换为你自己拍摄的图像路径）
    image_paths = [
        "campus_road1.jpg",  # 替换为你的图像路径
        "campus_road2.jpg",  # 替换为你的图像路径
    ]
    
    for image_path in image_paths:
        print(f"处理图像: {image_path}")
        
        try:
            # 处理图像
            result, edges, roi_edges = lane_detector.process_image(image_path)
            
            if result is not None:
                # 读取原图用于显示
                original = cv2.imread(image_path)
                
                # 可视化结果
                lane_detector.visualize_results(original, edges, roi_edges, result, image_path)
            else:
                print(f"处理失败: {image_path}")
                
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")

if __name__ == "__main__":
    main()

