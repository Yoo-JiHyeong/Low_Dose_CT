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
    """
    Get optimal threshold by iterating
    """
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
    """
    Get lung location

    apply the threshold
    -> remove components which touch the border
    -> get two largest components (actually, three components including the bed)

    """
    def _identify_lung(ind_img):
        res_img = np.zeros_like(ind_img, dtype=bool)
        threshold = get_threshold(ind_img)

        # thresholding
        res_img[ind_img > threshold] = False
        res_img[ind_img <= threshold] = True

        # remove components touching the border of the image
        border_cleared_image = segmentation.clear_border(res_img, bgval=False)

        # labeling connected components
        labeled_components = measure.label(border_cleared_image)
        areas = [r for r in measure.regionprops(labeled_components)]

        # get the largest components
        areas.sort(key=lambda r: r.area)
        if len(areas) > 2:
            for region in areas[:-3]:
                for coordinates in region.coords:
                    labeled_components[coordinates[0], coordinates[1]] = 0

        # remove bed
        areas.sort(key=lambda r: r.centroid)
        for coordinates in areas[-1].coords:
            labeled_components[coordinates[0], coordinates[1]] = 0

        border_cleared_image = labeled_components > 0

        return border_cleared_image

    if len(image.shape) == 3:
        return np.stack([_identify_lung(bc_image) for bc_image in image])
    else:
        return _identify_lung(image)


def lung_component_3d(image, get_bound=True):
    """
    by finding two largest 3d connected components, get only lung component from whole body
    if needed, return upper_bound and lower bound of lung region (z-axis)
    """

    lung_upper_bound = 0
    lung_lower_bound = 0

    res_image = np.zeros_like(image, dtype=bool)

    lung_image = identify_lung(image)

    labeled_component = measure.label(lung_image)
    areas_3d = measure.regionprops(labeled_component)
    areas_3d.sort(key = lambda r: r.area)

    for area in areas_3d[-2:]:
        print(area.area)
        lung_upper_bound = np.amax(area.coords, 0)[0]
        lung_lower_bound = np.amin(area.coords, 0)[0]
        for coord in area.coords:
            res_image[coord[0], coord[1], coord[2]] = True

    if get_bound:
        return res_image, lung_upper_bound, lung_lower_bound
    else:
        return res_image


def get_mediastinum_mask_v1(image):
    """
    "Automated mediastinal lymph node detection from CT volumes based on intensity targeted"
    Oda et al, 2017
    """
    # get lung location
    lung_image, upper_bound, lower_bound = lung_component_3d(image)
    lung_index = [[np.where(x[:-1] != x[1:]) for x in ind_image] for ind_image in lung_image[lower_bound:upper_bound]]

    mediastinum_mask = np.zeros_like(image, dtype=bool)

    # fill the area between lung with 'True'
    for z_index, z in enumerate(lung_index):
        for y_index, y in enumerate(z):
            if len(y[0][::2]) >= 2:
                for x_index in range(len(y[0][::2]) - 1):
                    x1 = y[0][2*x_index+1]
                    x2 = y[0][2*x_index+2]
                    mediastinum_mask[z_index, y_index, x1:x2+1] = True

    return mediastinum_mask, upper_bound, lower_bound


def get_mediastinum(image, mask_func):
    # apply mediastinum mask to the original image
    mediastinum_image = np.copy(image)
    mediastinum_mask, upper_bound, lower_bound = mask_func(image)
    # print(np.count_nonzero(mediastinum_mask))
    # print("-------------------")
    for z_index, z in enumerate(mediastinum_mask):
        labeled_area = measure.label(z)
        region_props = measure.regionprops(labeled_area)
        region_props.sort(key=lambda r: r.area)
        # print(np.count_nonzero(mediastinum_mask))
        # print([r.area for r in region_props])
        for region in region_props:
            if region.area < 0.01 * image.shape[1] * image.shape[2]:
                # remove the region, if area of the region is smaller than 1% of image volume
                for coord in region.coords:
                    mediastinum_mask[z_index, coord[0], coord[1]] = False

    mediastinum_image[mediastinum_mask != True] = -2000

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


        lung_3d = lung_component_3d(npys)

        lung_3d.astype(int)
        # plt.subplot(121)
        # plt.imshow(lung_3d[151], cmap='gray')
        # plt.subplot(122)
        # plt.imshow(lung_3d[120], cmap='gray')
        # plt.show()


        utils.Plot2DSlice(lung_3d[50:200])

        # tmp_res = get_mediastinum(npys[-111:-62], get_mediastinum_mask_v1)
        # tmp_res2 = identify_lung(npys)
        # tmp_res = get_mediastinum(npys, get_mediastinum_mask_v1)
        # utils.Plot2DSlice(tmp_res2)
        # utils.Plot2DSlice(npys[-111:-62])


        # for creating segmented dicom file ... extremely slow / need a compliment
        # loaded = utils.load_3d_dcm(r"C:\Users\yjh36\Desktop\TMT LAB\FDG-PET2\10918160_20120912_105713_PT")
        # for i, item in enumerate(loaded[-111:-62]):
        #
        #     tmp_res[i] += 1024
        #     tmp_res[i].clip(0,None,tmp_res[i])
        #     tmp_res[i].astype("uint16")
        #     for n, val in enumerate(item.pixel_array.flat):
        #         item.pixel_array.flat[n] = tmp_res[i].flat[n]
        #
        #     item.PixelData = item.pixel_array.tostring()
        #
        #     item.save_as("C:/Users/yjh36/Desktop/sample/" + str(i) + ".dcm")



        """
        tmp = npys[-90]
        tmp2 = identify_lung(tmp)
        plt.subplot(121)
        plt.imshow(tmp, cmap='gray')
        plt.subplot(122)
        plt.imshow(tmp2, cmap='gray')
        plt.show()
        """