from torch.utils.data import Dataset
from tqdm import tqdm
import os
from glob import glob
import torch
import numpy as np
from utils.face_preprocess import face_preprocess, calculate_dct
import cv2
from torch_dct import idct_2d

class Res_Dataset(Dataset):
    def __init__(self, cfg):
        self.n_frames = cfg["num_frames"]
        self.n_file = cfg["num_file"]
        self.real_path = os.path.join(cfg["dataset_path"], "original_sequences", "youtube", cfg["comp"], "frames")
        self.fake_path = os.path.join(cfg["dataset_path"], "manipulated_sequences", cfg["dataset"], cfg["comp"], "frames")
        self.real_list = []
        self.fake_list = []
        for i in tqdm(range(self.n_file)):
            img_real = glob(os.path.join(self.real_path, str(i).zfill(3), '*'))
            img_fake = glob(os.path.join(glob(os.path.join(self.fake_path, str(i).zfill(3) + '*'))[0], '*'))

            real_path = os.path.join(cfg["dataset_path"], "dct_data", "real_dct", str(i).zfill(3))
            fake_path = os.path.join(cfg["dataset_path"], "dct_data", "fake_dct", str(i).zfill(3))

            for j in range(self.n_frames):
                if os.path.exists(real_path + "/" + str(j).zfill(3) + ".pt"):
                    pass
                else:
                    real_img, fake_img, sbi_img, landmarks, sbi_landmarks = face_preprocess(img_real[j], img_fake[j], cfg["image_size"])
                    # cv2.imwrite(real_path + "/" + str(j).zfill(3) + ".png", real_img)
                    os.makedirs(real_path, exist_ok=True)
                    torch.save(calculate_dct(real_img / 255), os.path.join(real_path, str(j).zfill(3) + ".pt"))

                    # cv2.imwrite(fake_path + "/" + str(j).zfill(3) + ".png", fake_img)
                    os.makedirs(fake_path, exist_ok=True)
                    torch.save(calculate_dct(fake_img / 255), os.path.join(fake_path, str(j).zfill(3) + ".pt"))

                self.real_list.append(os.path.join(real_path, str(j).zfill(3) + ".pt"))
                self.fake_list.append(os.path.join(fake_path, str(j).zfill(3) + ".pt"))

    def __len__(self):
        return len(self.real_list)

    def __getitem__(self, idx):
        real = torch.load(self.real_list[idx]).to(torch.float32)
        fake = torch.load(self.fake_list[idx]).to(torch.float32)
        real = torch.clip(idct_2d(real, norm='ortho'), 0, 1)
        fake = torch.clip(idct_2d(fake, norm='ortho'), 0, 1)

        return real, fake


    def collate_fn(self, batch):
        img_r, img_f = map(torch.stack, zip(*batch))
        data = {}
        data['img'] = torch.cat([img_r.float(), img_f.float()], 0)
        data['label'] = torch.tensor([0] * len(img_r) + [1] * len(img_f))
        return data

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)




