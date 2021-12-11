import os
import cv2
import numpy as np
from paddle.vision.transforms import Transpose, Normalize

image_normalize = Normalize(mean=[127.5],std=[127.5]) #去均值化
image_transpose = Transpose() #图片维度转换


def read_images(data_path, images_path):
    out_np_imaegs = []
    for image_path in images_path:
        image_path = os.path.join(data_path,image_path)
        paddle_tensor = image_normalize(cv2.imread(image_path))
        paddle_tensor = image_transpose(paddle_tensor)
        out_np_imaegs.append(paddle_tensor)
    return np.array(out_np_imaegs)

def read_labels(data_path, labels_path):
    out_np_labels = []
    for label_path in labels_path:
        label_path = os.path.join(data_path, label_path)
        label_array = cv2.imread(label_path)[:,:,0]
        paddle_tensor = image_transpose(label_array)
        out_np_labels.append(paddle_tensor)
    return np.array(out_np_labels)