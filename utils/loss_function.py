import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from net.mobileface.infer import Predictor, args_mobileface
import torch.autograd as autograd
# from .dct import idct_2d
from torch_dct import idct_2d
from torchvision.transforms import Resize
from time import time
from PIL import Image
import os


predictor = Predictor(args_mobileface.mtcnn_model_path, args_mobileface.mobilefacenet_model_path,
                      args_mobileface.face_bank_path)

device = torch.device("cuda")
torch_resize = Resize([380, 380], antialias=True)
torch_resize_resnet = Resize([224, 224], antialias=True)

def tensor_matrix(size, low, up):
    matrix = torch.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if up > i + j >= low:
                matrix[i][j] = 1.
    return matrix

def dct_mask(dct, mask):
    img = torch.clip(idct_2d(dct * mask, norm='ortho') * 255, 0, 255)
    # a = Image.fromarray(np.uint8(np.array(img[0].permute(1,2,0).detach().cpu())))
    # a.show()
    return img.permute(0, 2, 3, 1)


def PriorAndIntegrityLoss(M1, M2, M3, cfg):
    size = cfg["image_size"]
    low_boundary = cfg["low_boundary"]
    high_boundary = cfg["high_boundary"]

    M1_standard = tensor_matrix(size, 0, low_boundary).expand(M1.shape).to(device)
    M2_standard = tensor_matrix(size, low_boundary, high_boundary).expand(M1.shape).to(device)
    M3_standard = tensor_matrix(size, high_boundary, size * 2).expand(M1.shape).to(device)
    M3_ones = tensor_matrix(size, 0, size * 2).expand(M1.shape).to(device)

    Mseloss = nn.MSELoss()
    M1_loss = Mseloss(M1, M1_standard)*(size**2/(low_boundary**2/2))/10
    M2_loss = Mseloss(M2, M2_standard)*(size**2/(size**2-(low_boundary**2/2)-((size*2-high_boundary)**2/2)))
    M3_loss = Mseloss(M3, M3_standard)*(size**2/((size*2-high_boundary)**2/2))
    M_loss = Mseloss(M1+M2+M3, M3_ones)
    return M1_loss + M2_loss + M3_loss + M_loss

def AuthentictyDeterminativeloss(M1, M2, M3, real, fake, resnet):
    image_real_M1 = dct_mask(real, M1).permute(0, 3, 1, 2)
    image_fake_M1 = dct_mask(fake, M1).permute(0, 3, 1, 2)
    image_real_M1M2 = dct_mask(real, M1+M2).permute(0, 3, 1, 2)

    image_fake_M1M2 = dct_mask(fake, M1+M2).permute(0, 3, 1, 2)
    synthetic_fake_M1M2 = dct_mask(real*M1+fake*M2, 1).permute(0, 3, 1, 2)

    # synthetic_final_M1M2 = dct_mask(fake*M1+real*M2, 1).permute(0, 3, 1, 2)

    loss1 = sum(resnet(torch_resize_resnet(image_real_M1)/255).softmax(1)[:, 1])+\
            sum(resnet(torch_resize_resnet(image_fake_M1)/255).softmax(1)[:,1])+\
            sum(resnet(torch_resize_resnet(image_real_M1M2) / 255).softmax(1)[:, 1])

    loss2 = sum(1-resnet(torch_resize_resnet(image_fake_M1M2)/255).softmax(1)[:, 1])+\
            sum(1-resnet(torch_resize_resnet(synthetic_fake_M1M2)/255).softmax(1)[:, 1])

    return (loss1 + loss2)/(2*len(M1))


def Cosine_Similarity(feature, feature1):
    result = 0
    for i in range(len(feature)):
        prob = torch.clip(torch.dot(feature[i], feature1[i]) / (torch.linalg.norm(feature[i]) * torch.linalg.norm(feature1[i])), 0, 1)
        result += (1 - prob)
    return result/len(feature)


def FacialFidelityLoss(M1, M2, M3, real, fake, label, fake_label, landmarks):

    M1_real = dct_mask(real, M1)
    M1_fake = dct_mask(fake, M1)
    real_feature, real_feature_bank = predictor.recognition(M1_real, landmarks, label)
    fake_feature, fake_feature_bank = predictor.recognition(M1_fake, landmarks, fake_label)

    loss = (Cosine_Similarity(real_feature, real_feature_bank)+Cosine_Similarity(fake_feature, fake_feature_bank))

    return loss

def QualityAgnosticLoss(M1, M2, M3, real, fake):
    real_image = dct_mask(real, 1)
    M1_real = dct_mask(real, M1)
    image_synthetic_M1M2 = dct_mask(real*M1+fake*M2, 1)

    MseLoss = nn.MSELoss()
    loss = 4 * MseLoss(real_image, image_synthetic_M1M2)
    loss += MseLoss(M1_real, image_synthetic_M1M2)

    return loss


def loss_function(M1, M2, M3, real, fake, resnet, additional_data, cfg=None):
    weights = cfg["lambda"]
    labels, fake_labels, landmarks = additional_data

    FF_loss = FacialFidelityLoss(M1, M2, M3, real, fake, labels, fake_labels, landmarks)

    AD_loss = AuthentictyDeterminativeloss(M1, M2, M3, real, fake, resnet)

    QA_loss = QualityAgnosticLoss(M1, M2, M3, real, fake)

    PI_loss = PriorAndIntegrityLoss(M1, M2, M3, cfg["PILossBoundary"])

    return weights["lambda1"]*FF_loss, weights["lambda2"]*AD_loss, weights["lambda3"]*QA_loss, weights["lambda4"]*PI_loss

