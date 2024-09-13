import numpy as np
from plantcv import plantcv as pcv
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from pilkit.lib import Image
from pilkit.processors import TrimBorderColor

import os

#plant cv Documentation

# https://github.com/danforthcenter/plantcv/tree/main/docs


# Set global debug behavior to None (default), "print" (to file),
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "print"

# def read_image(image_path):
#     #Read image
#     img = Image.open(image_path)
#
#     #Transform to numpy array
#     np_img = np.asarray(img)
#     # img_name, img_extension = os.path.splitext(image_path)



# def creating_mask(image_path):
#     #Read image
#     img = Image.open(image_path)
#
#     #Transform to numpy array
#     np_img = np.asarray(img)
#     # img_name, img_extension = os.path.splitext(image_path)
#
#     # Label grid of seeds using ROIs
#     grid_rois = pcv.roi.multi(img=np_img, coord=(31, 31), radius=20, spacing=(67, 67), nrows=4, ncols=7)
#     labeled_mask, num_seeds = pcv.create_labels(mask=clean_mask, rois=grid_rois, roi_type="partial")
#
#     # Don't use ROIs but instead assume one "object of interest" per contour
#     labeled_mask2, num_seeds2 = pcv.create_labels(mask=clean_mask)
def transform_image(image_path):

    # I can read image also like this:
    #Read image
    # img = Image.open(image_path)

    #Transform to numpy array
    # np_img = np.asarray(img)

    #But this is less code
    # read in image
    np_img, path, img_filename = pcv.readimage(filename=image_path, mode="native")
    print(path)
    print(img_filename)

    # Optionally, set a sample label name
    # pcv.params.sample_label = "plant"

    # # 1 Original
    # pcv.print_image(np_img, img_filename)
    # pcv.print_image(np_img, "/Users/air/Documents/ecole/leaffliction/original_image.png")


    # #2 Gaussian blur DONE
    # # Apply gaussian blur to a binary image that has been previously thresholded.
    # gaussian_img = pcv.gaussian_blur(img=np_img, ksize=(51, 51), sigma_x=0, sigma_y=None)
    # # img_flip.save(f"{img_name}_Flip{img_extension}")
    # # pcv.print_image(gaussian_img, "/Users/air/Documents/ecole/leaffliction/gaussian_image.png")


    #2_1 Thresholded image from gray
    # image converted from RGB to gray.
    gray_img = pcv.rgb2gray(rgb_img=np_img)
    # Create binary image from a gray image based on threshold values,
    # targeting light objects in the image.
    threshold_light = pcv.threshold.binary(gray_img=gray_img, threshold=80, object_type='dark', max_value = 255)
    # centers, optimal_radius_size = auto_roi(labeled_image)

    # 2_2 Thresholded image from LAB
    # Blue-Yellow ('b') channel is output
    b_channel = pcv.rgb2gray_lab(rgb_img=np_img, channel='a')

    # Create masked image from a color image based LAB color-space and threshold values.
    # for lower and upper_thresh list as: thresh = [L_thresh, A_thresh, B_thresh]
    mask, masked_img = pcv.threshold.custom_range(img=np_img, lower_thresh=[0, 0, 100], upper_thresh=[255, 255, 255],channel='LAB')

    #3 Mask
    # Apply binary 'white' mask over an image.
    masked_image = pcv.apply_mask(img=np_img, mask=threshold_light, mask_color='white')


    # # Run naive bayes multiclass and save a list of masks
    # mask = pcv.naive_bayes_classifier(np_img, pdf_file="/machine_learning.txt")
    # # Plot each class with it's own color
    # # plotted = pcv.visualize.colorize_masks(
    # #     masks=[mask['plant'], mask['pustule'], mask['background'], mask['chlorosis']],
    # #     colors=['green', 'red', 'gray', 'gold'])


    #4 Roi objects (Region of interest to mask)


    # # Identify objects
    # # find_objects(img, mask, device, debug=None)
    # new_img = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)
    # id_objects, obj_hierarchy = pcv.find_objects(img = np_img, mask = masked_image)



    # Make a grid of ROIs
    center_x = int(np_img.shape[0]/2)
    print(center_x)
    center_y = int(np_img.shape[1] / 2)
    print(center_y)
    radius = int((center_x + center_y )/2)
    print(radius)

    roi = pcv.roi.circle(img=np_img, x=center_x, y=center_y, r=75)

    roi1 = pcv.roi.from_binary_image(bin_img=threshold_light, img=np_img)
    roi_contour, roi_hierarchy = pcv.roi.from_binary_image(bin_img=threshold_light, img=np_img)

    # Analyze the shape of a plant
    # shape_img = pcv.analyze_object(img=np_img, labeled_mask=roi, n_labels=1, label="plant")
#!!!
    # shape_header, shape_data, shape_img = pcv.analyze_object(img=np_img, objects, mask =roi, "/home/malia/setaria_shape_img.png")

    # roi = pcv.roi.from_binary_image(bin_img=threshold_light, img=np_img)

    # # Decide which objects to keep
    # roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy, id_objects,obj_hierarchy)


    # ROI filter allows the user to define if objects partially inside ROI are included or if objects are cut to ROI.
    # filtered_mask = pcv.roi.filter(mask=threshold_light, roi=roi, roi_type='partial')


    # # Create rects representing the image and the ROI
    # # Make a grid of ROIs
    #
    # # Don't use ROIs but instead assume one "object of interest" per contour
    # # labeled_mask2, num_seeds2 = pcv.create_labels(mask=clean_mask)
    #
    # device, roi1, roi_hierarchy = pcv.define_roi(masked2, 'rectangle', device, None, 'default', debug, True, 550, 0,  -500, -1900)
    #
    # rois = pcv.roi.auto_grid(nrows=1, ncols=1, img=np_img)
    #
    # # Convert the ROI contour into a binary mask
    # mask = pcv.roi.roi2mask(img=img, roi=rois)

    # # ROI objects allows the user to define if objects partially inside ROI are included or if objects are cut to ROI.
    # roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img, roi, roi_hierarchy, objects, obj_hierarchy, 'partial')

    #
    # # roi = cv::Rect(-50, 50, 200, 100)
    # # roi = pcv.roi.custom(img=np_img, vertices=[[1190,490], [1470,830], [920,1430], [890,950]])
    # # # Convert the ROI contour into a binary mask
    # # mask = pcv.roi.roi2mask(img=img, roi=roi)
    # roi_img = pcv.analyze.size(img=np_img, labeled_mask=mask, n_labels=1)
    # # Save returned images with more specific naming
    # pcv.print_image(roi_img, "/Users/air/Documents/ecole/leaffliction/roi_image.png")


    # #5 Analyze object - Analyze plant shape

    # analysis_image = pcv.analyze.size(img=np_img, labeled_mask=roi)


    # # Characterize object shapes
    # # shape_img = pcv.analyze.size(img=np_img, labeled_mask=mask_fill, label="default")
    # shape_img = pcv.analyze.size(img=np_img, labeled_mask=mask, n_labels=1)
    # # Save returned images with more specific naming
    # pcv.print_image(shape_img, "/Users/air/Documents/ecole/leaffliction/shape_image.png")
    # # Access data stored out from analyze.size
    # plant_solidity = pcv.outputs.observations['plant_1']['solidity']['value']
    # color_histogram = pcv.analyze.color(rgb_img=np_img, labeled_mask=None, colorspaces='all', label="default")

    # #6 Pseudolandmarks
    # device = 1
    # # Identify a set of land mark points
    # # Results in set of point values that may indicate tip points
    # device, top, bottom, center_v = pcv.x_axis_pseudolandmarks(obj, mask, np_img, device, debug='print')


    # 7 Histogram Analyze Color DONE
    # hist = pcv.visualize.histogram(img=np_img)
    pcv.params.debug = "print"
    #color_header, color_data, \
    analysis_images = pcv.analyze_color(np_img, None) #, 256, None, 'v', 'img', "analyze_color.png")
    # color_histogram = pcv.analyze.color(rgb_img=np_img, labeled_mask=None, colorspaces='hsv', label="default")


    #This saves results for one image, and each image is saved individually if you run another image (it will overwrite the last one)
    # pcv.outputs.save_results(filename= args.result)

    # Create a foreground mask from both images






if __name__ == "__main__":
    import sys

    directory = "/Users/air/Documents/ecole/leaffliction/images/Apple_Black_rot/image (1).JPG"
    # Apply transformation
    transform_image(directory)
