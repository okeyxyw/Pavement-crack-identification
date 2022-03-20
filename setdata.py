#!/user/bin/python
# coding=utf-8
from os.path import isdir
from scipy import io
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    file_path = 'CrackForest-dataset-master/CrackForest-dataset-master/groundTruth/'
    png_img_dir = 'CrackForest-dataset-master/CrackForest-dataset-master/groundTruthPngImg/'
    if not isdir(png_img_dir):
        os.makedirs(png_img_dir)
    image_path_lists = os.listdir(file_path)
    images_path = []
    for index in range(len(image_path_lists)):
        image_file = os.path.join(file_path, image_path_lists[index])
        images_path.append(image_file)
        image_mat = io.loadmat(image_file)
        segmentation_image = image_mat['groundTruth']['Segmentation'][0]
        segmentation_image_array = np.array(segmentation_image[0])
        image = Image.fromarray((segmentation_image_array -1) * 255)
        png_image_path = os.path.join(png_img_dir, "%s.png" % image_path_lists[index][0:3])
        image.save(png_image_path)
        plt.figure()
        plt.imshow(image)
        # plt.pause(2)
        plt.show()
