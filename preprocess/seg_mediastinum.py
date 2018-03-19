from preprocess import utils
from glob import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_mediastinum(original_image, lung_seg_image):
    lung_mask = np.zeros_like(lung_seg_image, dtype=bool)

    lung_mask[lung_seg_image != 0] = True

    original_image[lung_mask] = -2000

    return original_image

def generate_markers(image):
    def _generate_marker(img):
        # Creation of the internal Marker
        marker_internal = img < -400
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)
        # print(*np.unique(marker_internal_labels, return_counts=True))
        areas = [r for r in measure.regionprops(marker_internal_labels)]
        areas.sort(key=lambda r: r.area)
        # print([r.area for r in areas])
        if len(areas) > 2:
            for region in areas[:-3]:
                # print(region.area, areas[-1].area)
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
        """
        print(areas)
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0
        """
        marker_internal = marker_internal_labels > 0
        # Creation of the external Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a
        # Creation of the Watershed Marker matrix
        marker_watershed = np.zeros((512, 512), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        return marker_internal, marker_external, marker_watershed

    print(image.shape[1:])
    if len(image.shape) >= 3:
        internal_stacks, external_stacks, watershed_stacks = list(), list(), list()
        for r in image:
            gm = _generate_marker(r)
            internal_stacks.append(gm[0])
            external_stacks.append(gm[1])
            watershed_stacks.append(gm[2])

        internal_stacks = np.stack(internal_stacks)
        external_stacks = np.stack(external_stacks)
        watershed_stacks = np.stack(watershed_stacks)

        return internal_stacks, external_stacks, watershed_stacks
    else:
        return _generate_marker(image)


def seperate_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))

    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed


if __name__ == "__main__":

    for folder in glob(r"C:\Users\yjh36\Desktop\TMT LAB\FDG-PET2\*"):
        print(folder + " is projected")
        dcm_path = folder

        # dcms = load_3d_dcm(dcm_path, image_type="CT")
        # dcms = load_3d_dcm(dcm_path, image_type="PET")
        dcms = utils.load_3d_dcm(dcm_path, parsetype="CT")
        npys = utils.dcm_to_npy(dcms)
        print(npys.shape)

        tmp = npys[-90]

        # plt.subplot(221)
        # plt.imshow(tmp, cmap='gray')

        # print("Original Image")
        #
        # test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(tmp)
        test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(npys)
        # test_patient_internal = generate_markers(npys)
        print("Internal Marker")
        # print(test_patient_internal.shape)
        utils.Plot2DSlice(test_patient_internal)
        # # plt.subplot(222)
        # # plt.imshow(test_patient_internal, cmap='gray')
        # print("External Marker")
        # # plt.subplot(223)
        # # plt.imshow(test_patient_external, cmap='gray')
        # utils.Plot2DSlice(test_patient_external)
        # print("Watershed Marker")
        # # plt.subplot(224)
        # # plt.imshow(test_patient_watershed, cmap='gray')
        # # plt.show()
        # utils.Plot2DSlice(test_patient_watershed)

        # Some Testcode:
        # test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, test_marker_internal, test_marker_external, test_marker_watershed = seperate_lungs(
        #     tmp)
        #
        # plt.imshow(get_mediastinum(tmp, test_marker_internal), cmap='gray')
        # plt.show()
        # print("Sobel Gradient")
        # plt.imshow(test_sobel_gradient, cmap='gray')
        # plt.show()
        # print("Watershed Image")
        # plt.imshow(test_watershed, cmap='gray')
        # plt.show()
        # print("Outline after reinclusion")
        # plt.imshow(test_outline, cmap='gray')
        # plt.show()
        # print("Lungfilter after closing")
        # plt.imshow(test_lungfilter, cmap='gray')
        # plt.show()
        # print("Segmented Lung")
        # plt.imshow(test_segmented, cmap='gray')
        # plt.show()
