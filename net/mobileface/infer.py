import argparse
import functools
import os
import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image
from .detection.face_detect import MTCNN
from .utils.utils import add_arguments, print_arguments
from skimage import transform as trans
import torch.nn.functional as F
from .models.arcmargin import ArcNet
from .models.mobilefacenet import MobileFaceNet
from skimage.transform._geometric import _umeyama as get_sym_mat
from torchvision.transforms import Resize
from utils.funcs import load_json
from tqdm import tqdm

mobileface_path = load_json("config/base.json")["mobileface_path"]
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('face_bank_path', str, os.path.join(mobileface_path, 'mobileface', 'face_bank'), 'facebank path')
add_arg('mobilefacenet_model_path', str,
        os.path.join(mobileface_path, 'mobileface', 'save_model', 'mobilefacenet.pth'),
        'MobileFaceNet model path')
add_arg('mtcnn_model_path', str, os.path.join(mobileface_path, 'mobileface', 'save_model', 'mtcnn'),
        'MTCNN model path')
args_mobileface = parser.parse_args()
print_arguments(args_mobileface)


class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_bank_path, threshold=0.7):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)
        self.device = torch.device("cuda")
        self.state_dict = torch.jit.load(mobilefacenet_model_path).state_dict()
        self.model = MobileFaceNet()
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.faces_bank = self.load_face_bank(face_bank_path)

    def load_face_bank(self, face_bank_path):
        print('loading face_bank images...')
        faces_bank = {}
        for path in tqdm(os.listdir(face_bank_path)):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_bank_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img)
            if imgs is None or len(imgs) > 1:
                print('picture %s in face library contains not one face, automatically skip the picture' % image_path)
                continue
            imgs = self.process(imgs)
            feature = self.infer(imgs[0])
            faces_bank[name] = feature[0][0]
        return faces_bank

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    def tensor_process(self, imgs):
        img = imgs.permute((2, 0, 1))
        img = (img - 127.5) / 127.5
        return img

    def norm_crop(self, img, landmark, image_size=112):
        M = self.estimate_norm(landmark)
        theta = self.convert_to_theta_cv2(M, img, image_size)
        warped = self.my_warpAffine(img, theta, image_size, borderValue=0.0)
        return warped

    def estimate_norm(self, lmk):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        src = np.array([[38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)
        M = get_sym_mat(lmk, src, estimate_scale=True)
        return M

    def convert_to_theta_cv2(self, M, img, image_size=112):
        H, W, _ = img.shape
        M = np.linalg.inv(M)
        A = np.array([[2 / W, 0, -1],
                      [0, 2 / H, -1],
                      [0, 0, 1]])
        C = np.array([[2 / image_size, 0, -1],
                      [0, 2 / image_size, -1],
                      [0, 0, 1]])
        theta = A @ M @ np.linalg.inv(C)
        return theta[:2, :]

    def my_warpAffine(self, img, theta, image_size, borderValue):
        H, W, _ = img.size()
        theta = torch.tensor(theta, dtype=torch.float32).view(1, 2, 3)
        theta = theta.to('cuda')
        grid = F.affine_grid(theta, torch.Size([1, 3, image_size, image_size]), align_corners=True)
        output = F.grid_sample(img.permute(2, 0, 1).unsqueeze(0), grid, padding_mode='zeros', align_corners=True,
                               mode='bicubic')
        output = torch.clamp(output, 0, 255)
        return output.squeeze(0).permute(1, 2, 0)

    def infer(self, imgs):
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        features = []
        for i in range(imgs.shape[0]):
            img = imgs[i][np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float32, device=self.device)

            feature = self.model(img)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features

    def tensor_infer(self, imgs):
        imgs = imgs.unsqueeze(0)
        with torch.enable_grad():
            feature = self.model(imgs)
        return feature

    def estimate_eye_center(self, eye_landmarks):
        eye_landmarks = np.array(eye_landmarks)
        eye_center = np.mean(eye_landmarks, axis=0)
        return eye_center

    def my_landmarks(self, landmarks):
        landmarks = landmarks.numpy()
        result_landmarks = np.array(
            [self.estimate_eye_center(landmarks[36:42]), self.estimate_eye_center(landmarks[42:48]),
             landmarks[30], landmarks[48], landmarks[54]])
        return result_landmarks

    def recognition(self, image, landmarks, label):
        feature_list = torch.tensor([]).to('cuda')
        feature_bank_list = torch.tensor([]).to('cuda')
        for i in range(len(image)):
            landmark = self.my_landmarks(landmarks[i])
            imgs = self.norm_crop(image[i], landmark)
            imgs = self.tensor_process(imgs)  # 归一化
            features = self.tensor_infer(imgs)  # features
            feature = features[0]

            name = str(np.array(label[i])).zfill(3)
            feature_bank = torch.tensor(self.faces_bank[name]).to(self.device)
            feature_list = torch.cat([feature_list, feature.unsqueeze(0)])
            feature_bank_list = torch.cat([feature_bank_list, feature_bank.unsqueeze(0)])

        return feature_list, feature_bank_list
        #     prob = torch.clip(torch.dot(feature, feature1) / (torch.linalg.norm(feature) * torch.linalg.norm(feature1)), 0, 1)
        #     probs = torch.cat([probs, prob.unsqueeze(0)])
        # return probs


    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype('simfang.ttf', size)
        draw.text((left, top), text, color)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_face(self, image, boxes_c, names):
        img = image
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                img = self.add_text(img, name, corpbbox[0], corpbbox[1] - 15, color=(0, 0, 255), size=12)
        cv2.imshow("result", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    predictor = Predictor(args_mobileface.mtcnn_model_path, args_mobileface.mobilefacenet_model_path,
                          args_mobileface.face_bank_path, threshold=args_mobileface.threshold)
    # boxes, names = predictor.recognition(args_mobileface.image_path)
    # predictor.draw_face(args_mobileface.image_path, boxes, names)
