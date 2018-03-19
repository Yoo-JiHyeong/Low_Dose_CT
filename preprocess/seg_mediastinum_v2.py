from preprocess import utils
from glob import glob
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_threshold(image, prev_threshold=None):
    if prev_threshold is None:
        body_mean = image[image > 0].mean()
        nonbody_mean = -1000
    else:
        body_mean = image[image >= prev_threshold].mean()
        nonbody_mean = image[image < prev_threshold].mean()

    threshold = (body_mean + nonbody_mean) / 2

    if prev_threshold == threshold:
        return threshold
    else:
        return get_threshold(image, threshold)


def identify_lung(image):
    def _identify_lung(ind_img):
        threshold = get_threshold(ind_img)

        res_img = np.zeros_like(ind_img, dtype=bool)
        # thresholding
        res_img[ind_img > threshold] = False
        res_img[ind_img <= threshold] = True

        # remove components touching the border of the image
        border_cleared_image = segmentation.clear_border(res_img, bgval=False)

        # labeling connected components
        labeled_components = measure.label(border_cleared_image)
        areas = [r for r in measure.regionprops(labeled_components)]

        minimum_volume = res_img.shape[0] * res_img.shape[1] * 0.01

        areas.sort(key=lambda r: r.area)
        if len(areas) > 2:
            for region in areas[:-3]:
                # print(region.area, areas[-1].area)
                for coordinates in region.coords:
                    labeled_components[coordinates[0], coordinates[1]] = 0

        border_cleared_image = labeled_components > 0

        return border_cleared_image
    if len(image.shape) == 3:
        return np.stack([_identify_lung(bc_image) for bc_image in image])
    else:
        return _identify_lung(image)

def get_mediastinum_mask_v1(image):
    """
    "Automated mediastinal lymph node detection from CT volumes based on intensity targeted"
    Oda et al, 2017
    """

    image = identify_lung(image)
    lung_index = [[np.where(x[:-1] != x[1:]) for x in ind_image] for ind_image in image]

    mediastinum_mask = np.zeros_like(image, dtype=bool)

    for z_index, z in enumerate(lung_index):
        for y_index, y in enumerate(z):
            # print(y)
            # print(y[0][::2])
            if len(y[0][::2]) >= 2:
                for x_index in range(len(y[0][::2]) - 1):
                    # print(y[0][2*x_index+1], y[0][2*x_index+2])
                    x1 = y[0][2*x_index+1]
                    x2 = y[0][2*x_index+2]
                    mediastinum_mask[z_index, y_index, x1:x2+1] = True

    # remove bed from mask
    for z in mediastinum_mask:
        labeled_area = measure.label(z)
        area_props = measure.regionprops(labeled_area)
        area_props.sort(key=lambda r : r.area)
        print([r.area for r in area_props])
        area1, area2 = area_props[-1], area_props[-2]
        if area1.centroid[0] > area2.centroid[0]:
            for coord in area1:
                labeled_area[coord[0], coord[1]] = False
        else:
            for coord in area2:
                labeled_area[coord[0], coord[1]] = False

    return mediastinum_mask


def get_mediastinum(image, mask_func):
    mediastinum_image = np.copy(image)
    mask = mask_func(image)
    mediastinum_image[mask != True] = -2000

    return mediastinum_image

if __name__ == "__main__":

    # for test
    for folder in glob(r"C:\Users\yjh36\Desktop\TMT LAB\FDG-PET2\*"):
        print("loaded : " + folder)
        dcm_path = folder
        dcms = glob(folder + "/*.dcm")

        dcms = utils.load_3d_dcm(dcm_path, parsetype="CT")
        npys = utils.dcm_to_npy(dcms)

        if npys.shape[0] > 300:
            continue
        print("shape : " + str(npys.shape))

        tmp_res = get_mediastinum(npys[-111:-62], get_mediastinum_mask_v1)
        utils.Plot2DSlice(tmp_res)
        # utils.Plot2DSlice(npys[-111:-62])

        """
        tmp = npys[-90]
        tmp2 = identify_lung(tmp)
        plt.subplot(121)
        plt.imshow(tmp, cmap='gray')
        plt.subplot(122)
        plt.imshow(tmp2, cmap='gray')
        plt.show()
        """