import numpy as np
import cv2

def compute_optical_flow_farneback(img1, img2):
    """计算两张图像的光流，使用 Farneback 方法。

    Args:
        img1: 第一张图像。
        img2: 第二张图像。

    Returns:
        光流向量场。
    """

    # 将图像转换为灰度图。
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 设置 Farneback 光流估计方法的参数。
    params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # 计算光流。
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **params)

    return flow

def motion_change(flow):
    # 计算光流场的运动变化程度
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    motion_change = np.mean(magnitude)

    return motion_change

if __name__ == "__main__":

    # 读取两张图像。
    img1 = cv2.imread("path/to/img1.jpg")
    img2 = cv2.imread("path/to/img2.jpg")

    # 计算光流。
    flow = compute_optical_flow_farneback(img1, img2)
    change = motion_change(flow)
