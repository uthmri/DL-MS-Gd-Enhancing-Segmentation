import os
import sys
import math
import shutil
import pickle
import numpy as np
import nibabel as nib
from keras.models import load_model
from keras import optimizers as opt
from skimage.measure import label, regionprops
from metrics import weighted_dice_coefficient_loss, dice_coefficient, precision, recall, get_lr_metric, tissue_dice, tissue_accuracy
from mri_functions import load_cases_from_csv, get_immediate_subdirectories, format_final_label
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from alt_generator import data_extraction_from_files


def visualization(img_resolution, images_folder, model, test_images, patch_res):
    # function to iterate network through images and predict segmentation

    # loop to iterate through images in given set
    for img_idx, img_name in enumerate(test_images):

        # source directory where images are contained
        dir_to_move = "{0}/{1}/".format(images_folder, img_name)

        # destionation directory where network outputs and original files will be stored
        dir_destination = 'outputs_test/{0}/'.format(img_name)

        # prints loop progress
        print("Image: {0} //  {1} out of {2}".format(img_name , img_idx + 1, len(test_images)))

        # loads MRI data from files and ground truth labels
        data_array, label_array = data_extraction_from_files(images_folder, img_name)

        # buffer array to store MRI information in array form
        buff_array = np.ndarray((data_array.shape[0], data_array.shape[1], data_array.shape[2]))


        # step definion to iterate through network axes
        iter_x, iter_y, iter_z = math.ceil( data_array.shape[0]/patch_res[0] ), math.ceil( data_array.shape[1]/patch_res[1] ), math.ceil( data_array.shape[2]/patch_res[2])


        # loops to iterate thorugh network axes
        for x_step in range(iter_x):
            for y_step in range(iter_y):
                for z_step in range(iter_z):

                    # coordinate definition to pass network to the whole volume, network takes 3D patches from volume
                    x_coord_init, x_coord_end = x_step*patch_res[0], x_step*patch_res[0] + patch_res[0]
                    y_coord_init, y_coord_end = y_step*patch_res[1], y_step*patch_res[1] + patch_res[1]
                    z_coord_init, z_coord_end = z_step*patch_res[2], z_step*patch_res[2] + patch_res[2]

                    # checks to ensure network patches are within image axes
                    if x_coord_end > img_resolution[0]:
                        x_coord_init -= ( x_coord_end - data_array.shape[0] )
                        x_coord_end = img_resolution[0]

                    if y_coord_end > img_resolution[1]:
                        y_coord_init -= ( y_coord_end - data_array.shape[1] )
                        y_coord_end = img_resolution[1]

                    if z_coord_end > img_resolution[2]:
                        z_coord_init -= ( z_coord_end - data_array.shape[2] )
                        z_coord_end = img_resolution[2]


                    # creates array to store data to be predicted
                    image_to_predict = data_array[x_coord_init:x_coord_end, y_coord_init:y_coord_end, z_coord_init:z_coord_end]

                    # reshapes array into network expected format
                    image_to_predict = image_to_predict.reshape(1, patch_res[0], patch_res[1], patch_res[2], 5)

                    # network predicts segmentation on given patch array
                    predicted_array = model.predict(image_to_predict)

                    # network ouputs come in the form of scores 0 to 1 for each class, this step assigns the most likely class to the corresponding voxel
                    predicted_array = np.argmax(predicted_array, -1)

                    # saves predicted patch segmentation into corresponding voxels in full image array
                    buff_array[x_coord_init:x_coord_end, y_coord_init:y_coord_end, z_coord_init:z_coord_end] = predicted_array[0 ,: ,: ,:]


        print("Class tissues in network output: {0}".format(np.unique(buff_array)))

        # reference image to set affine file for all volumes
        ref_img_path = '{0}/{1}/validated.nii.gz'.format(images_folder, test_images[0] )
        ref_img = nib.load(ref_img_path)
        affine_set = ref_img.affine


        # creates nifti file to which network prediction will be saved
        new_image = nib.Nifti1Image(buff_array, affine=affine_set)

        # assings name to nifti file
        nifti_name = "outputs_test/{0}/network_new.nii.gz".format( img_name )


        # copies data from original data folder to folder containing outputs for easier handling of data
        if os.path.exists(dir_destination) == False:

            shutil.copytree(dir_to_move, dir_destination)

        # saves nifti image containing network predictions
        nib.save(new_image, nifti_name)



def main():

    # folder which contains images cases
    image_folder = "../Gad_Baseline"

    # file path to network saved model
    unet_folder = "model_gad_3d_baseline.hdf5"

    # folder which contains information on image cases
    lists_folder = "cases_lists_files"

    # general resolution of patient cases available
    image_resolution = (256, 256, 44)




    test_folders_names_file = lists_folder + "/" + "test" + ".csv"


    folders_path = load_cases_from_csv(test_folders_names_file)


    actual_list = folders_path.copy()


    whole_set_files = get_immediate_subdirectories(image_folder)
    for case in actual_list:
        if case not in whole_set_files:
            folders_path.remove(case)





    # network parameters required for compilation
    learn_rate = 1e-4
    optimizer_A = opt.SGD(lr=learn_rate, nesterov=True)
    lr_metric = get_lr_metric(optimizer_A)

    # Model is loaded along with previously saved metrics
    model_to_save = load_model(unet_folder,
                               custom_objects={'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss
                                   , 'dice_coefficient': dice_coefficient, 'recall': recall, 'precision': precision
                                   , "background_dice": tissue_dice(0, "background_dice")
                                   ,"grey_dice": tissue_dice(1, "grey_dice"), "white_dice": tissue_dice(2, "white_dice")
                                   , "csf_dice": tissue_dice(3, "csf_dice"), "lesion_dice": tissue_dice(4, "lesion_dice")
                                   , "gad_dice": tissue_dice(5, "gad_dice"), 'lr': lr_metric, 'InstanceNormalization': InstanceNormalization})

    # model is compiled and ready to predict
    model_to_save.compile(loss=weighted_dice_coefficient_loss, optimizer=optimizer_A, metrics=['categorical_accuracy'
        , 'categorical_crossentropy', dice_coefficient, recall, precision, tissue_dice(0, "background_dice")
        , tissue_dice(1, "grey_dice"), tissue_dice(2, "white_dice"), tissue_dice(3, "csf_dice"), tissue_dice(4, "lesion_dice")
        , tissue_dice(5, "gad_dice"), lr_metric])


    # model is run to set of testing images
    visualization(img_resolution= image_resolution, images_folder=image_folder, model=model_to_save, test_images=folders_path, patch_res=(128, 128, 16))



if __name__ == "__main__":
    main()
