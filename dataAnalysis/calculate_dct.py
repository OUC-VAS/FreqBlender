import os

import argparse
from numpy.fft import fft2, fftshift
import cv2
import numpy as np
from energy_contrast import face_cut
from glob import glob
from PIL import Image


# def calculate_psd(img):
#     img = cv2.resize(img, (380, 380))
#     fd_Im = fftshift(fft2(img))
#     psd = abs(fd_Im) ** 2
#     # psd = 10 * np.log10(psd)
#     return psd


def calculate_dct(img):
    img = cv2.resize(img, (380, 380))
    dct_img = cv2.dct(img.astype(np.float32))
    return dct_img


def face_preprocess(real_path, fake_path):
    landmark = np.load(real_path.replace('.png', '.npy').replace('frames', 'landmarks'))[0]
    real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    fake_img = cv2.imread(fake_path, cv2.IMREAD_GRAYSCALE)
    return face_cut(real_img, landmark)[0].astype(np.float32), face_cut(fake_img, landmark)[0].astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset',
                        choices=['Deepfakes', 'FaceSwap', "Face2Face", 'NeuralTextures'], default='Deepfakes')
    parser.add_argument('-p', dest='dataset_path')
    parser.add_argument('-c', dest='comp', choices=['raw', 'c23', 'c23'], default='c23')
    args = parser.parse_args()
    ## power spectral
    real_path = os.path.join(args.dataset_path, 'original_sequences', 'youtube', args.comp, 'frames')
    fake_path = os.path.join(args.dataset_path, 'manipulated_sequences', args.dataset, args.comp, 'frames')

    real_path_list = []
    fake_path_list = []

    max_count = 3
    real_img = sorted(glob(real_path + '*'))
    for i in real_img:
        counter = 0
        for j in sorted(glob(i + '/*')):
            if os.path.exists(glob(j.replace('original_sequences', 'manipulated_sequences')
                                           .replace('youtube', args.dataset)
                                           .replace(os.path.basename(i), os.path.basename(i) + '*'))[0]):
                real_path_list.append(j)
                fake_path_list.append(glob(j.replace('original_sequences', 'manipulated_sequences')
                                           .replace('youtube', args.dataset)
                                           .replace(os.path.basename(i), os.path.basename(i) + '*'))[0])
                counter += 1
                if counter >= max_count:
                    break

    counter_real = np.zeros((380, 380))
    counter_fake = np.zeros((380, 380))

    for i in range(len(real_path_list)):
        real_img, fake_img = face_preprocess(real_path_list[i], fake_path_list[i])

        real_img = calculate_dct(real_img)
        fake_img = calculate_dct(fake_img)
        counter_real += real_img
        counter_fake += fake_img

    os.makedirs("dct_statistics", exist_ok=True)
    np.save(os.path.join("dct_statistics", args.dataset + '_real.npy'), counter_real)
    np.save(os.path.join("dct_statistics", args.dataset + '_fake.npy'), counter_fake)
