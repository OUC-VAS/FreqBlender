import os
import sys
import numpy as np
from .energy_contrast import face_cut
from utils import blend as B
from .funcs import RandomDownScale
import albumentations as alb
import logging
import cv2
import torch
from torch_dct import dct_2d

if os.path.isfile('.utils/library/bi_online_generation.py'):
    print('exist library')
    exist_bi = True
else:
    print('library load failed...')
    exist_bi = False

if exist_bi:
    from utils.library.bi_online_generation import random_get_hull


def face_preprocess(real_path, fake_path, image_size):
    landmark = np.load(real_path.replace('.png', '.npy').replace('frames', 'landmarks'))[0]

    real_img = cv2.cvtColor(cv2.imread(real_path), cv2.COLOR_BGR2RGB)
    fake_img = cv2.cvtColor(cv2.imread(fake_path), cv2.COLOR_BGR2RGB)
    # sbi_img, _, sbi_landmark = sbi_image(real_img.copy(), landmark)

    real_tmp, real_landmark = face_cut(real_img, landmark, image_size)
    fake_tmp, fake_landmark = face_cut(fake_img, landmark, image_size)
    # sbi_tmp, sbi_landmark = face_cut(sbi_img, sbi_landmark, image_size)
    # return np.array(real_tmp), np.array(fake_tmp), np.array(sbi_tmp), real_landmark, sbi_landmark
    return np.array(real_tmp), np.array(fake_tmp),  real_landmark



def sbi_image(img, landmark, phase="train"):
    landmark = reorder_landmark(landmark)
    if phase == 'train':
        if np.random.rand() < 0.5:
            img, landmark = hflip(img, landmark)

    img_r, img_f, mask_f = self_blending(img.copy(), landmark.copy())
    if phase == 'train':
        transformed = get_transforms()(image=img_f.astype('uint8'), image1=img_r.astype('uint8'),
                                       get_transforms=get_transforms())
        img_f = transformed['image']
        img_r = transformed['image1']
    return img_f, img_r, landmark


def calculate_dct(img):
    dct_tensor = dct_2d(torch.tensor(img).permute(2, 0, 1), norm='ortho')
    return dct_tensor


def face_cut_retina(img, bbox):
    H, W = len(img), len(img[0])
    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    w = x1 - x0
    h = y1 - y0
    w0_margin = w / 4
    w1_margin = w / 4
    h0_margin = h / 4
    h1_margin = h / 4

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)
    img_cropped = img[y0_new:y1_new, x0_new:x1_new]

    return img_cropped


def get_source_transforms():
    return alb.Compose([
        alb.Compose([
            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=1),
            alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
        ], p=1),

        alb.OneOf([
            RandomDownScale(p=1),
            alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        ], p=1),

    ], p=1.)


def get_transforms():
    return alb.Compose([

        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                               val_shift_limit=(-0.3, 0.3), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
        alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

    ],
        additional_targets={f'image1': 'image'},
        p=1.)


def randaffine(img, mask):
    f = alb.Affine(
        translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
        scale=[0.95, 1 / 0.95],
        fit_output=False,
        p=1)

    g = alb.ElasticTransform(
        alpha=50,
        sigma=7,
        alpha_affine=0,
        p=1,
    )

    transformed = f(image=img, mask=mask)
    img = transformed['image']
    mask = transformed['mask']

    transformed = g(image=img, mask=mask)
    mask = transformed['mask']

    return img, mask


def self_blending(img, landmark):
    H, W = len(img), len(img[0])
    if np.random.rand() < 0.25:
        landmark = landmark[:68]
    if exist_bi:
        logging.disable(logging.FATAL)
        mask = random_get_hull(landmark, img)[:, :, 0]
        logging.disable(logging.NOTSET)
    else:
        mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

    source = img.copy()
    if np.random.rand() < 0.5:
        source = get_source_transforms()(image=source.astype(np.uint8), )['image']
    else:
        img = get_source_transforms()(image=img.astype(np.uint8))['image']

    source, mask = randaffine(source, mask)

    img_blended, mask = B.dynamic_blend(source, img, mask)
    img_blended = img_blended.astype(np.uint8)
    img = img.astype(np.uint8)

    return img, img_blended, mask


def reorder_landmark(landmark):
    landmark_add = np.zeros((13, 2))
    for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark


def hflip(img, landmark=None):
    H, W = img.shape[:2]
    landmark = landmark.copy()

    if landmark is not None:
        landmark_new = np.zeros_like(landmark)

        landmark_new[:17] = landmark[:17][::-1]
        landmark_new[17:27] = landmark[17:27][::-1]

        landmark_new[27:31] = landmark[27:31]
        landmark_new[31:36] = landmark[31:36][::-1]

        landmark_new[36:40] = landmark[42:46][::-1]
        landmark_new[40:42] = landmark[46:48][::-1]

        landmark_new[42:46] = landmark[36:40][::-1]
        landmark_new[46:48] = landmark[40:42][::-1]

        landmark_new[48:55] = landmark[48:55][::-1]
        landmark_new[55:60] = landmark[55:60][::-1]

        landmark_new[60:65] = landmark[60:65][::-1]
        landmark_new[65:68] = landmark[65:68][::-1]
        if len(landmark) == 68:
            pass
        elif len(landmark) == 81:
            landmark_new[68:81] = landmark[68:81][::-1]
        else:
            raise NotImplementedError
        landmark_new[:, 0] = W - landmark_new[:, 0]

    else:
        landmark_new = None

    img = img[:, ::-1].copy()
    return img, landmark_new
