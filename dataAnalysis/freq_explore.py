import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import cv2
import radialProfile
from scipy.interpolate import griddata
import argparse
import os

def transition(magnitude_spectrum):
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    points = np.linspace(0, len(psd1D), num=psd1D.size)
    xi = np.linspace(0, len(psd1D), num=len(psd1D))
    interpolated = griddata(points, psd1D, xi, method='cubic')
    interpolated = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))
    psd1D_org = interpolated
    return psd1D_org


def dataPrint(dataset, dataset_path):
    fshift_real = np.load(dataset_path + '_real.npy')
    fshift_fake = np.load(dataset_path + '_fake.npy')
    magnitude_spectrum_real = np.log(np.abs(fshift_real) + 1)  # Amplitude spectrum
    magnitude_spectrum_fake = np.log(np.abs(fshift_fake) + 1)  # Amplitude spectrum

    psd1D_org_real = transition(magnitude_spectrum_real)
    psd1D_org_fake = transition(magnitude_spectrum_fake)

    x = np.arange(0, len(psd1D_org_real), 1)
    plt.plot(x[:], psd1D_org_real[:], '#2878B5', label='real')
    plt.plot(x[:], psd1D_org_fake[:], '#C82423', label='fake')
    plt.legend()
    plt.title(dataset)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', choices=['Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures'])
    args=parser.parse_args()
    dataset_path = os.path.join("dct_statistics", args.dataset)
    dataPrint(args.dataset, dataset_path)

