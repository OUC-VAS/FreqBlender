import cv2
import numpy as np
import math
import os

np.set_printoptions(suppress=True)


def face_cut(img, landmark, size=400):
    H, W = len(img), len(img[0])

    x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
    x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
    w = x1 - x0
    h = y1 - y0

    w0_margin = w / 8
    w1_margin = w / 8
    h0_margin = h / 8
    h1_margin = h / 8

    w0_margin *= (np.random.rand() * 0.6 + 0.2)
    w1_margin *= (np.random.rand() * 0.6 + 0.2)
    h0_margin *= (np.random.rand() * 0.6 + 0.2)
    h1_margin *= (np.random.rand() * 0.6 + 0.2)


    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin)+1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin)+1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new, :]

    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]

    img_final = cv2.resize(img_cropped, (size, size))
    landmark_cropped[:, 0] = landmark_cropped[:, 0]*(size/len(img_cropped[0]))
    landmark_cropped[:, 1] = landmark_cropped[:, 1]*(size/len(img_cropped))
    return img_final, landmark_cropped


def get_face(img, filename):
    if os.path.isfile(filename.replace('.png', '.npy').replace('/frames/', '/landmarks/')):
        landmark = np.load(filename.replace('.png', '.npy').replace('/frames/', '/landmarks/'))[0]
        img, landmark = face_cut(img, landmark)
    else:
        return np.array([])

    return img


def extract_dct(img, size):
    img = cv2.resize(img, (size, size))

    count = []
    for j in range(0, 3):
        y = img[:, :, j]
        y = y.astype(np.float32)
        y = cv2.dct(y)
        count.append(y)
    tmp = np.concatenate(count).reshape(3, size, size)
    return tmp


def energy_statistic(data):
    count = []
    for i in range(3):
        tmp = band_statistics(data[i])
        count.append(tmp)
    return count


def band_statistics(data):
    count = [0] * 8
    for j in range(8):
        counter = 0
        for m in range(400):
            for n in range(400):
                if m + n >= j * 50 and m + n < (j + 1) * 50:
                    counter += 1
                    count[j] += (data[m][n]) ** 2
        count[j] /= counter
    return count


# if __name__ == '__main__':
#     real_name = '/media/lhz/DATA/dataset/FF++/original_sequences/youtube/c23/frames/000/000.png'
#     fake_name = '/media/lhz/DATA/dataset/FF++/manipulated_sequences/Deepfakes/c23/frames/000_003/000.png'
#
#     real_img = cv2.imread(real_name, 1)
#     fake_img = cv2.imread(fake_name, 1)
#
#     real_img = get_face(real_img, real_name)
#     fake_img = get_face(fake_img, fake_name)
#
#     cv2.imshow('real', real_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imshow('fake', fake_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     real_img = extract_dct(real_img, size=400)
#     fake_img = extract_dct(fake_img, size=400)
#
#     fake_img = energy_statistic(fake_img)
#     real_img = energy_statistic(real_img)
#     print(np.array(fake_img[0]) + np.array(fake_img[1]) + np.array(fake_img[2]) - np.array(real_img[0]) - np.array(
#         real_img[1]) - np.array(real_img[2]))
