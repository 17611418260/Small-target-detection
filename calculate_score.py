import sys
sys.path.insert(1,'/data/workspace/myshixun/project')
sys.path.insert(1,'/home/headless/Desktop/workspace/myshixun/project')
import os
import numpy as np
from model_predict import predict
from tqdm import tqdm
from utilss.get_map import test_map

model = predict()
classes_path = 'data_classes.txt'
val_annotation_path = 'test.txt'
threshold = 0.4

with open(val_annotation_path, encoding='utf-8') as f:
    val_lines = f.read().split('\n')


def box_iou(b1, b2):
    b1_mins = np.array([b1[..., 0], b1[..., 1]])
    b1_maxes = np.array([b1[..., 2], b1[..., 3]])
    b1_wh = b1_maxes - b1_mins

    b2_mins = np.array([b2[:, 1], b2[:, 0]]).T
    b2_maxes = np.array([b2[:, 3], b2[:, 2]]).T
    b2_wh = b2_maxes - b2_mins

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, np.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area

    iou = intersect_area / union_area
    return iou

def career_college(lines):
    score_list = []
    for i in tqdm(lines):
        line = i.split(' ')

        if line[0].lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):

            results = model.detect_image(line[0])

            num_shape = 5
            if results is not None and len(results) != 0:

                true_like = np.array([j.split(',') for j in line[1:]], dtype='int32')

                top_like = np.array(results)

                true_label = np.array(true_like[:, 4], dtype='int32')
                true_conf = np.ones((num_shape,))
                true_boxes = true_like[:, :4]

                top_label = np.array(top_like[:, 6], dtype='int32')
                top_conf = top_like[:, 4] * top_like[:, 5]
                top_boxes = np.array(top_like[:, :4], dtype='int32')

                sum_iou = []
                sum_conf = []
                for num, box in enumerate(true_boxes):
                    box_zeros = np.zeros((num_shape, 4))
                    box_zeros[:len(box)] = box

                    iou = box_iou(box, top_boxes)
                    max_index = np.argmax(iou)
                    if iou[max_index] > threshold and top_label[max_index] == true_label[num]:
                        sum_conf.append(top_conf[max_index])
                    else:
                        sum_conf.append(0)
                    sum_iou.append(iou[max_index])

                sum_iou = np.array(sum_iou)
                sum_conf = np.array(sum_conf)

                score = (np.mean(sum_iou) + np.mean(sum_conf)) / 2
                score_list.append(score)

            else:
                score_list.append(0)

    score_list = np.array(score_list)
    scores = np.mean(score_list)
    return scores


# scores = career_college(val_lines)
# print(scores)


def regular_college():

    # 遍历所有文件，搜寻最大的文件，为模型文件
    max_size = 0
    for i, j, files in os.walk("project"):

        for f in files:
            path_f = os.path.join(i, f)
            f_size = os.path.getsize(path_f)
            if max_size < f_size:
                max_size = f_size

    # 计算 mAP 与图像检测速度
    mAP, speed = test_map(val_lines, model, classes_path)

    return mAP, str(max_size / 1000 ** 2), speed * 1000


if __name__ == '__main__':

    mAP, ModelSize, Speed = regular_college()
    print(f"mAP：{mAP},", end=' ')
    print(f"ModelSize: {ModelSize} M, Speed: {Speed} ms/image")