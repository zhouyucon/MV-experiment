# 实验配置参数
class Config:
    SOBEL_X_KERNEL = [[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]]
    
    SOBEL_Y_KERNEL = [[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]
    
    CUSTOM_KERNEL = [[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]]
    
    RESULTS_DIR = './results/'
    HISTOGRAM_BINS = 256