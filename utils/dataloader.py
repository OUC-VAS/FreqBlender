import os
from glob import glob
import cv2
from tqdm import tqdm
import numpy as np
import time
import torch
from torch.utils.data import Dataset
from .face_preprocess import face_preprocess, calculate_dct, sbi_image


class dataloader(Dataset):
    def __init__(self, n_frames=8, n_file=1000, cfg=None):
        print('\nloading data...\n')
        self.real_path = cfg["real_path"]
        self.fake_path = cfg["fake_path"]
        self.label_list = []
        self.real_list = []
        self.fake_list = []
        self.sbi_list = []
        self.landmark_list = []
        self.sbi_landmark_list = []
        self.wrong_face = [281, 604]
        for i in tqdm(range(n_file)):
            input_num = 0
            img_real = glob(os.path.join(self.real_path, str(i).zfill(3), '*'))
            img_fake = glob(os.path.join(glob(os.path.join(self.fake_path, str(i).zfill(3) + '*'))[0], '*'))

            real_path = os.path.join(cfg["dct_path"], "real_dct", str(i).zfill(3))
            fake_path = os.path.join(cfg["dct_path"], "fake_dct", str(i).zfill(3))
            # sbi_path = os.path.join(cfg["dct_path"], "sbi_dct", str(i).zfill(3))
            landmark_path = os.path.join(cfg["dct_path"], "landmark", str(i).zfill(3))
            # sbi_landmark_path = os.path.join(cfg["dct_path"], "sbi_landmark", str(i).zfill(3))

            try:
                for j in range(n_frames):
                    if os.path.exists(os.path.join(real_path, str(j).zfill(3) + ".pt")):
                        input_num += 1
                    else:
                        # Until we find the landmark image that exists
                        while os.path.exists(img_real[input_num].replace('.png', '.npy').replace('frames', 'landmarks')):
                            break
                        else:
                            input_num += 1

                        # real_img, fake_img, sbi_img, landmarks, sbi_landmarks = face_preprocess(img_real[input_num], img_fake[input_num], cfg["image_size"])
                        real_img, fake_img, landmarks = face_preprocess(img_real[input_num], img_fake[input_num], cfg["image_size"])

                        # cv2.imwrite(os.path.join(real_path, str(j).zfill(3) + ".png"), real_img)
                        os.makedirs(real_path, exist_ok=True)
                        torch.save(calculate_dct(real_img / 255), os.path.join(real_path, str(j).zfill(3) + ".pt"))

                        # cv2.imwrite(os.path.join(sbi_path, str(j).zfill(3) + ".png"), sbi_img)
                        # os.makedirs(sbi_path, exist_ok=True)
                        # torch.save(calculate_dct(sbi_img / 255), os.path.join(sbi_path, str(j).zfill(3) + ".pt"))

                        # cv2.imwrite(os.path.join(fake_path, str(j).zfill(3) + ".png"), fake_img)
                        os.makedirs(fake_path, exist_ok=True)
                        torch.save(calculate_dct(fake_img / 255), os.path.join(fake_path, str(j).zfill(3) + ".pt"))

                        os.makedirs(landmark_path, exist_ok=True)
                        np.save(os.path.join(landmark_path, str(j).zfill(3) + ".npy"), landmarks)

                        # os.makedirs(sbi_landmark_path, exist_ok=True)
                        # np.save(os.path.join(sbi_landmark_path, str(j).zfill(3) + ".npy"), sbi_landmarks)

                    self.real_list.append(os.path.join(real_path, str(j).zfill(3) + ".pt"))
                    self.fake_list.append(os.path.join(fake_path, str(j).zfill(3) + ".pt"))
                    # self.sbi_list.append(os.path.join(sbi_path, str(j).zfill(3) + ".pt"))
                    self.landmark_list.append(os.path.join(landmark_path, str(j).zfill(3) + ".npy"))
                    # self.sbi_landmark_list.append(os.path.join(sbi_landmark_path, str(j).zfill(3) + ".npy"))

                    labels = os.path.basename(os.path.dirname(img_fake[j]))
                    self.label_list.append([int(labels[:3]), int(labels[-3:])])
            except:
                print('video {} have some troubles'.format(i))

    def __len__(self):
        return len(self.real_list)

    def __getitem__(self, idx):
        if idx // 8 in self.wrong_face:
            idx = torch.randint(low=0, high=len(self), size=(1,)).item()

        real = torch.load(self.real_list[idx]).to(torch.float32)
        fake = torch.load(self.fake_list[idx]).to(torch.float32)
        landmarks = np.load(self.landmark_list[idx])
        # sbi = torch.load(self.sbi_list[idx]).to(torch.float32)
        # sbi_landmarks = np.load(self.sbi_landmark_list[idx])
        # return real, fake, sbi, (self.label_list[idx][0], self.label_list[idx][1], landmarks, sbi_landmarks)
        return real, fake, (self.label_list[idx][0], self.label_list[idx][1], landmarks)
