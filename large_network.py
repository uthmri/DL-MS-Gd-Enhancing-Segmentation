import sys

# Library used for loading and saving python objects to files
import pickle


from math import floor

# Function used to shuffle data in random/semi-random basis
from random import shuffle

from keras import backend as K

# Imports functions and variables to handle keras models
from keras.models import Model, load_model

# Imports functions for dealing with keras optimizers
import keras.optimizers as opt

# imports model from network architecture
from net_architecture import i_model

# function to log network metrics after each training iteration
from keras.callbacks import CSVLogger

# function to run model training on multiple GPUs
from keras.utils import multi_gpu_model

# function thats runs network through a set of images to predict segmentation
from volume_total import visualization

# useful for passing several images through network during training
from alt_generator import DataGenerator

# loss function which weights classes equally
from metrics import weighted_dice_coefficient_loss, tissue_dice, tissue_accuracy, dice_coefficient, get_lr_metric, recall, precision

# function to save network model aftere iteration upon certain conditions
from keras.callbacks import ModelCheckpoint

# this library contains helpful functions to deal with csv lists and nifti images
from mri_functions import load_cases_from_csv, get_immediate_subdirectories, save_list_to_csv

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


# saves python object to pickle file
def save_obj( obj, name, obj_folder ):
    with open(obj_folder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# custom multi-gpu class to integrate Keras multi_gpu_model class with ModelCheckpoint
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):

        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


# class that handles creation and training of neural network
class myUnet(object):


    # function to run network training upon certain parameters
    def train(self, csv_logger,  mode, imgs_train,
              imgs_validation, batch_size, patch_size, unet_folder, gpu_num, case_dimensions, case_folder):


        # sets of parameters for training set iterated through with DataGenerator
        params = {'dim':patch_size,
                  'case_dims':case_dimensions,
                  'cases_folder':case_folder,
                  'batch_size':batch_size,
                  'n_classes':6,
                  'n_channels':5,
                  'shuffle':True,
                  'augment':True}

        # sets of parameters for validation set iterated through with DataGenerator
        params_val = {'dim':patch_size,
                      'case_dims':case_dimensions,
                      'cases_folder':case_folder,
                      'batch_size':batch_size,
                      'n_classes':6,
                      'n_channels':5,
                      'shuffle':True,
                      'augment':False}



        train_set, val_set =[], []
        for idx, img_num in enumerate(imgs_train):
            train_set.append('Set' + str(img_num))

        for idx, img_num in enumerate( imgs_validation ):
            val_set.append('Set'+ str(img_num))


        partition = {'train':train_set, 'validation':val_set}
        labels = {'mask':['0']}


        # definition of DataGenerators to handle train and validation data
        train_image_generator = DataGenerator(partition['train'], labels, **params)
        val_image_generator = DataGenerator(partition['validation'], labels, **params_val)


        # learning to rate for network training
        learn_rate = 1e-5
        # definition of optimizer including asmgrad variation
        optimizer_A = opt.Adam(lr=learn_rate, amsgrad=True)
        # variable to obtain learning rate value in log file
        lr_metric = get_lr_metric(optimizer_A)

        # this mode sets network initial weights through Xavier initialization and then begins training
        if mode == 'create':

            # calls function to create network model
            model_to_save = i_model()

            # converts model into a multi-gpu model to be trained using multiple gpus
            parallel_model = ModelMGPU(model_to_save, gpu_num)

            # creates model checkpoint callback to save model for which lowest loss on validation data is obtained
            model_checkpoint = ModelCheckpoint(unet_folder, monitor='val_loss',
                                               verbose=1, save_best_only=True)

            # compiles model including several metrics and per-tissue dice scores
            parallel_model.compile(loss= weighted_dice_coefficient_loss , optimizer=optimizer_A, metrics=['categorical_accuracy'
                , 'categorical_crossentropy', dice_coefficient, recall, precision, tissue_dice(0, "background_dice")
                , tissue_dice(1, "grey_dice"), tissue_dice(2, "white_dice"), tissue_dice(3, "csf_dice"), tissue_dice(4, "lesion_dice")
                , tissue_dice(5, "gad_dice"), lr_metric])

            # fits model to data (training) for a certain number of iterations with end-of-epoch checks on validation data
            parallel_model.fit_generator(generator=train_image_generator, validation_data=val_image_generator,  use_multiprocessing=False
                                         , epochs=10, workers=16, callbacks=[csv_logger, model_checkpoint], verbose=2)


        # if this mode is used, then network it trained using its current weights contained in the specified .hdf5 file
        if mode == 'train':

            # metric to get learning value for which network is being trained with
            lr_metric = get_lr_metric(optimizer_A)

            # loads model from hdf5 file
            model_to_save = load_model(unet_folder, custom_objects={'weighted_dice_coefficient_loss':weighted_dice_coefficient_loss
                , 'dice_coefficient': dice_coefficient, 'recall': recall, 'precision': precision
                , "background_dice":tissue_dice(0, "background_dice"), "grey_dice":tissue_dice(1, "grey_dice")
                , "white_dice":tissue_dice(2, "white_dice"), "csf_dice":tissue_dice(3, "csf_dice"), "lesion_dice":tissue_dice(4, "lesion_dice")
                , "gad_dice":tissue_dice(5, "gad_dice"), 'lr':lr_metric, 'InstanceNormalization': InstanceNormalization})

            # creates model checkpoint callback to save model for which lowest loss on validation data is obtained
            model_checkpoint = ModelCheckpoint(unet_folder, monitor='val_loss',
                                               verbose=1, save_best_only=True)

            # converts model into a multi-gpu model to be trained using multiple gpus
            gpu_model = ModelMGPU(model_to_save, gpus=gpu_num)

            # compiles model including several metrics and per-tissue dice scores
            gpu_model.compile(loss= weighted_dice_coefficient_loss, optimizer=optimizer_A, metrics=['categorical_accuracy'
                , 'categorical_crossentropy', dice_coefficient, recall, precision, tissue_dice(0, "background_dice")
                , tissue_dice(1, "grey_dice"), tissue_dice(2, "white_dice"), tissue_dice(3, "csf_dice"), tissue_dice(4, "lesion_dice")
                , tissue_dice(5, "gad_dice"), lr_metric])

            # fits model to data (training) for a certain number of iterations with end-of-epoch checks on validation data
            gpu_model.fit_generator(train_image_generator, validation_data=val_image_generator, epochs=1000, verbose=2, use_multiprocessing=False
                                     ,workers=16, callbacks=[csv_logger, model_checkpoint])


# function to save training, validation, and testing sets lists in csv files
def list_save(lists_folder, lists_set, lists_set_names):

    for set_idx, set_list in enumerate(lists_set):
        save_list_to_csv(csv_file_name="{0}/{1}.csv".format(lists_folder, lists_set_names[set_idx])
                         , list_to_save=set_list)



if __name__ == '__main__':

    # clears GPUs from running processes
    K.clear_session()

    # folder containing image volumes
    image_folder = "../Gad_Baseline"

    # folder containing lists with training, validation, testing sets definitions
    lists_files_folder = "cases_lists_files"

    # sequences used to train network
    training_seq = ['flair',
                    'pd',
                    't1_pre',
                    't1_post',
                    't2']

    # list of files containing ground truth data
    validation_files = ['validated',
                      'gad']

    # number of MRI patches passing through network at each iteration
    batch_size = 4

    # number of GPUs available in machine
    gpu_num = 2

    # label values contained in ground truth
    label_values = [1, 2, 3, 4, 5]

    # slices to remove from top and bottom of images considered
    top_cut, bottom_cut = 2, 2

    # effective number of slices after removing top and bottom slices
    objective_slice_num = 44 - top_cut - bottom_cut

    # resolution of most of the images used in analysis
    image_resolution = (256, 256, 44)

    # patch size used for network training
    patch_size = (128, 128, 16)

    # file name for saving training log data
    logger_folder = "log_large.csv"

    # file names for lists containing baseline patients IDs and baseline patients IDs for patients presenting enhancement only
    file_list_cases, file_lists_cases_gad = "baseline.csv", "baseline_gad.csv"

    # file name for network model
    model_folder = "model_gad_3d_baseline.hdf5"

    # names of sets considered
    lists_sets_names = ["train", "val", "test"]

    mode_op = 'create'

    myunet = myUnet()


    csv_logger = CSVLogger(logger_folder, append=True, separator=';')








    gad_cases = get_immediate_subdirectories(image_folder)

    non_gad_baseline_cases = gad_cases.copy()
    gad_baseline_cases = load_cases_from_csv(lists_files_folder + "/" + file_lists_cases_gad)


    print(len(non_gad_baseline_cases))

    for case in gad_baseline_cases:
        if case in non_gad_baseline_cases:
            non_gad_baseline_cases.remove(case)

    print(len(non_gad_baseline_cases))
    print(len(gad_baseline_cases))

    shuffle(non_gad_baseline_cases)
    shuffle(gad_baseline_cases)



    train_part, val_part, test_part = 0.6, 0.2, 0.2

    num_non_gad_cases, num_gad_cases = len(non_gad_baseline_cases), len(gad_baseline_cases)

    print(num_non_gad_cases)
    print(num_gad_cases)



    train_set_non_gad, val_set_non_gad, test_set_non_gad = non_gad_baseline_cases[0:floor(num_non_gad_cases * train_part)]\
        , non_gad_baseline_cases[floor(num_non_gad_cases * train_part) : floor(num_non_gad_cases * (train_part + val_part) )]\
        ,  non_gad_baseline_cases[floor(num_non_gad_cases * (train_part + val_part)) : floor(floor(num_non_gad_cases * (train_part + val_part + test_part)))]

    train_set_gad, val_set_gad, test_set_gad = gad_baseline_cases[0:floor(num_gad_cases * train_part)] \
        , gad_baseline_cases[floor(num_gad_cases * train_part): floor(num_gad_cases * (train_part + val_part))] \
        , gad_baseline_cases[floor(num_gad_cases * (train_part + val_part)): floor(
        floor(num_gad_cases * (train_part + val_part + test_part)))]

    train_set, val_set, test_set = train_set_gad + train_set_non_gad, val_set_gad + val_set_non_gad, test_set_gad + test_set_non_gad

    print("Train: {0} | Val: {1} | Test: {2}".format(len(train_set), len(val_set), len(test_set)))



    if mode_op == "create":

        list_save(lists_folder=lists_files_folder, lists_set=[train_set, val_set, test_set], lists_set_names=lists_sets_names)

    elif mode_op == "train":

        train_list_csv, val_list_csv, test_list_csv = "{0}/{1}".format(lists_files_folder, "train.csv"), "{0}/{1}".format(lists_files_folder, "val.csv")\
            , "{0}/{1}".format(lists_files_folder, "test.csv")

        train_set, val_set, test_set = load_cases_from_csv(train_list_csv), load_cases_from_csv(val_list_csv), load_cases_from_csv(test_list_csv)



    # specify training, validation, and testing sets as lists of labels corresponding to available MRI cases number






    # calls for network training with specific parameters
    myunet.train(mode = mode_op, csv_logger = csv_logger, imgs_train = train_set
                 ,imgs_validation = val_set, batch_size = batch_size
                 , patch_size = patch_size, unet_folder = model_folder, gpu_num=gpu_num
                 , case_dimensions=image_resolution , case_folder=image_folder)

    # iterates network through test set images to predict possible segmentation
    visualization(img_resolution=image_resolution, images_folder=image_folder
                  , model=model_folder, test_images=test_set, patch_res=patch_size)


