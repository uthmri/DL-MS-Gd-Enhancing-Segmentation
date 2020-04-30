import os
import csv
import random
import pickle
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from keras.utils import multi_gpu_model
from keras.models import Model

# Auxiliar functions to process data

# Function to load pickle file into object
def load_obj( name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Saves object into pickle file
def save_obj( obj, name, obj_folder ):
    with open(obj_folder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Loads csv single column sequence of cases into a python list
def load_cases_from_csv(csv_file_name):

    base_list = []

    # opens specified .csv file
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # reads rows from csv file and appends entries to list
        for row in csv_reader:
            base_list.append(row[0])

    return base_list

# saves list to csv
def save_list_to_csv(csv_file_name, list_to_save):

    with open(csv_file_name, 'w+') as csv_file:
        for entry in list_to_save:
            csv_file.write(str(entry))
            csv_file.write("\n")

    csv_file.close()


# Gets immediate subdirectories for a specified main folder
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Class which allows to conveniently train a model on multiple GPUs and also to save checkpoints for that model
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


# aggregates labels for GM, WM, CSF, Lesion with labels for Gd-enhancing lesion
def format_final_label( image_resol, objective_slice_num, validated_array, gad_array, values ):

    # array tohold aggregation of labels
    final_lbl = np.zeros((image_resol[0], image_resol[1], image_resol[2], 1))

    # adjusts label array to effective image resolution
    validated_array = validated_array[:, :, 0:image_resol[2]]

    # removes undesired values from ground truth labels
    labels_mask = np.isin(validated_array, values)
    validated_array *= labels_mask

    # assigns value of 5 to all voxels in gad ground truth
    gad_array[ gad_array != 0 ] = 5

    # adjusts label array to effective image resolution
    gad_array = gad_array[: , :, 0:image_resol[2]]

    # inserts values in validated array where gad array has value equal to 0
    buff_arr = np.where(gad_array == 0, validated_array, gad_array)

    # assigns aggregate array to label array
    final_lbl[:, :, 0:image_resol[2], 0] = buff_arr


    return final_lbl



def main():
    pass



if __name__ == '__main__':
    main()

