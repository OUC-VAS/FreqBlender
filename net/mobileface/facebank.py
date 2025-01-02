import cv2
import numpy as np
import math
import os
from glob import glob
from tqdm import tqdm
np.set_printoptions(suppress=True)
import argparse


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def face_cut(img, landmark):
    H, W = len(img), len(img[0])

    x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
    x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
    w = x1 - x0  # dilb的宽度高度
    h = y1 - y0
    w0_margin = w / 8  # 0#np.random.rand()*(w/8)
    w1_margin = w / 8  # 缓冲带
    h0_margin = h / 8  # 0#np.random.rand()*(h/5) #这个2是个啥？？？？？
    h1_margin = h / 8

    # w0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
    # w1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
    # h0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
    # h1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()

    y0_new = max(0, int(y0 - h0_margin))  # 取img_box新的最小值且防止由于脸部取点过于靠近边缘导致margin大于剩余空间导致选取错误
    y1_new = min(H, int(y1 + h1_margin) + 1)  # 取img_box新的最大值且防止由于脸部取点过于靠近边缘导致margin大于于剩余空间导致选取错误
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]  # 取下裁减人脸
    if landmark is not None:  # 按照img_box中最小值对landmark进行偏移
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    return img_cropped, landmark_cropped


def face_cut_retina(img, bbox):
    H, W = len(img), len(img[0])
    x0, y0 = bbox[0]  # retina最小
    x1, y1 = bbox[1]  # retina最大
    w = x1 - x0  # retina的宽度，高度
    h = y1 - y0
    w0_margin = w / 4  # 0#np.random.rand()*(w/8)
    w1_margin = w / 4  # 缓冲带
    h0_margin = h / 4  # 0#np.random.rand()*(h/5)
    h1_margin = h / 4

    y0_new = max(0, int(y0 - h0_margin))  # 取img_box新的最小值且防止由于脸部取点过于靠近边缘导致margin大于剩余空间导致选取错误
    y1_new = min(H, int(y1 + h1_margin) + 1)  # 取img_box新的最大值且防止由于脸部取点过于靠近边缘导致margin大于于剩余空间导致选取错误
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)
    img_cropped = img[y0_new:y1_new, x0_new:x1_new]

    return img_cropped


retina_face = ['281', '604']


def get_face(img, filename):
    # 加载预处理的landmark
    if os.path.isfile(filename.replace('.png', '.npy').replace('/frames/', '/landmarks/')):
        landmark = np.load(filename.replace('.png', '.npy').replace('/frames/', '/landmarks/'))[0]
        # bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(),landmark[:, 1].max()])
        bboxes = np.load(filename.replace('.png', '.npy').replace('/frames/', '/retina/'))[:2]
        # Iou = IoUfrom2bboxes(bbox_lm, bboxes[0].flatten())

        if os.path.basename(os.path.dirname(filename)) not in retina_face:
            img, landmark = face_cut(img, landmark)
        else:
            img = face_cut_retina(img, bboxes[0])
    else:
        return np.array([])
    return img


if __name__ == '__main__':
    # dataset_path = "/media/lhz/Data/FF++"
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='dataset_path', default="/media/lhz/Data/FF++")
    args = parser.parse_args()

    path = os.path.join(args.dataset_path, 'original_sequences', 'youtube', 'c23', 'frames', '*')
    os.makedirs("face_bank", exist_ok=True)
    tmp_path = sorted(glob(path))
    for step, i in tqdm(enumerate(tmp_path), total=1000, desc="Processing"):
        # first image
        real_img = cv2.imread(os.path.join(i, '000.png'))
        real_img = get_face(real_img, os.path.join(i, '000.png'))
        real_img = cv2.resize(real_img, (400, 400))
        cv2.imwrite(os.path.join('face_bank', str(step).zfill(3) + '.png'), real_img)
