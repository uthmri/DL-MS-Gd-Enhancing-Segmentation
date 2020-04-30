import keras
import random
import pickle
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from mri_functions import format_final_label, load_obj


# function that loads MRI and ground truth data from files to numpy arrays
def data_extraction_from_files(dataset_path, img_folder, training_seq = ['flair', 'pd', 't1_pre', 't1_post', 't2']
                               , validation_seq = [ "validated", "gad" ], objective_slice_num = 44
                               , image_res = (256, 256, 50) , values = [0, 1, 2, 3, 4]):

    # buffer array to store arrays of MRI images
    final_img = np.zeros((image_res[0], image_res[1], image_res[2], len(training_seq)))

    # loads desired sequences into same array
    for sequence_idx, sequence_name in enumerate(sorted(training_seq)):

        current_img = nib.load(dataset_path + "/" + img_folder + "/" + sequence_name + ".nii.gz")
        current_img_data = current_img.get_data()
        current_img_data = current_img_data[:, :, :]

        final_img[:, :, 0:current_img_data.shape[-1], sequence_idx] = current_img_data

    final_img = final_img[:, :, 0:current_img_data.shape[-1], :]


    # buffer array to hold ground truth for data
    final_lbl = np.zeros((image_res[0], image_res[1], image_res[2], 1))
    current_lbl_data = None
    current_gad_data = None

    # loads ground truth data into label array
    for label_name in sorted(validation_seq):
        if label_name == 'validated':
            current_lbl = nib.load(dataset_path + "/" + img_folder + "/" + label_name + ".nii.gz")
            current_lbl_data = current_lbl.get_data()
            current_lbl_data = current_lbl_data[:, :, :]
            val_mask = np.isin(current_lbl_data, [0, 1, 2, 3, 4])
            current_lbl_data = current_lbl_data * val_mask

            if current_lbl_data.shape[2] > objective_slice_num:
                current_lbl_data = current_lbl_data[:, :, 0:objective_slice_num]

            final_lbl[:, :, 0:current_lbl_data.shape[2], 0] = current_lbl_data


        if label_name == 'gad':

            current_gad = nib.load(dataset_path + "/" + img_folder + "/" + label_name + ".nii.gz")
            current_gad_data = current_gad.get_data()
            current_gad_data = current_gad_data[:, :, :]


    # aggregates labels in ground truth file into single arrays
    if len(validation_seq) != 1:
        final_lbl = format_final_label( current_lbl_data.shape, objective_slice_num, current_lbl_data , current_gad_data, values)
    else:

        labels_mask = np.isin(final_lbl, values)
        final_lbl *= labels_mask

    # crops MRI array to remove zero arrays
    final_img = final_img[:, :, 0:current_img_data.shape[2], :]

    # normalizes values in MRI array
    final_img /= np.amax(final_img)

    # crops label array to remove zero valued arrays
    final_lbl = final_lbl[:, :, 0:current_img_data.shape[2], :]


    return final_img, final_lbl




# Function which implements cropping from volume
def crop_coords(patch_size, echo_image, val_image, default_size= (256, 256, 44), vol_name=None ):

    # selects random values from network
    random_selector = random.SystemRandom()
    gad_volume_check = False

    # reshapes validated data file for convinient processing
    buff_image = val_image.reshape((val_image.shape[0], val_image.shape[1], val_image.shape[2]))

    # masks tissues which are not T2 lesions from validated array
    gad_locations = np.isin(buff_image, 4)

    #
    vol_lesions = label(gad_locations, return_num=True, connectivity=2)

    # checks whether there are enhancing lesions in whole label array
    if 5 in np.unique(val_image):
        gad_volume_check = True



    check_count, gad_in = 0, False

    # loops until a patch containing Gd-enhancing lesion is obtained
    while check_count < 3 and gad_in == False:


        coord_set = []

        # loops through axes to pick up a random coordinate to crop patch from image volume array
        for dim_idx, dimension in enumerate(patch_size):

            random_coord = random_selector.choice(list(range( 0, default_size[dim_idx])) )

            range_max, range_min = random_coord + (dimension // 2), random_coord - (dimension // 2)

            # holds patch coordinates within image boundary
            if range_max > buff_image.shape[dim_idx]:
                val_op = range_max - buff_image.shape[dim_idx]
                range_max -= val_op
                range_min -= val_op

            elif range_min < 0:
                val_op = 0 - range_min
                range_max += val_op
                range_min += val_op

            coord_set.append([range_min, range_max])

        # crops randomly selected patch from label data
        y = val_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1], coord_set[2][0]:coord_set[2][1], :]


        # checks whether there is Gd-enhancing class in the identified patch
        if 5 in y:
            X = echo_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1], coord_set[2][0]:coord_set[2][1], :]
            gad_in = True

        # keeps looping if there is gad in whole volume or exits loop and moves to processing when there is no Gd-enhancing class in file
        elif gad_volume_check:
            check_count -= 1

        else:
            check_count += 1



    if gad_in == False:

        coord_set = []

        if len( regionprops(vol_lesions[0])) == 0:
            print("Empty {0}".format(vol_name))

        # selects a random T2 lesion within volume
        random_lesion = random_selector.choice(regionprops(vol_lesions[0]))

        # selects a random coordinate within all selected random lesion coordinates
        random_coord = random_selector.choice(random_lesion.coords)

        loop_criteria = False
        counter = 0

        # loops until a patch representative of all tissues of volume is obtained
        while loop_criteria == False:


            counter += 1

            # loops through patch dimension to assign random coordinates to guarantee that patch is within image array
            for dim_idx, dimension in enumerate(patch_size):

                # selects random values for which randomly selected coordinates would be shifted
                range_var = random_selector.choice(list(range(-dimension // 2, dimension // 2)))

                # shifts selected coordinates by random value
                range_max, range_min = random_coord[dim_idx] + (dimension // 2) + range_var, random_coord[dim_idx] - (dimension // 2) + range_var

                # holds patch coordinates within image boundary
                if range_max > buff_image.shape[dim_idx]:
                    val_op = range_max - buff_image.shape[dim_idx]
                    range_max -= val_op
                    range_min -= val_op

                elif range_min < 0:
                    val_op = 0 - range_min
                    range_max += val_op
                    range_min += val_op

                coord_set.append([range_min, range_max])

            # crops patches from original data according to random values
            X = echo_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1],
                    coord_set[2][0]:coord_set[2][1], :]

            y = val_image[coord_set[0][0]:coord_set[0][1], coord_set[1][0]:coord_set[1][1],
                    coord_set[2][0]:coord_set[2][1], :]


            # checks that all tissues are present in the obtained patch
            key_values = [0, 1, 2, 3, 4]
            result = all(elem in np.unique(y) for elem in key_values)


            if result:
                loop_criteria = True

            # empties set of coordinates
            coord_set = []


    return [X, y]





#Initialization of DataGenerator
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=2, dim=(256, 256, 32), n_channels=4,
                 n_classes=5, shuffle=True, augment=False, coordinate=False, cases_folder=None, case_dims=(256, 256 ,44)):
        'Initialization'
        self.dim = dim
        self.case_dims = case_dims
        self.cases_folder = cases_folder
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.coordinate = coordinate
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            ID = ID.split("t")[-1]

            # loads MRI and ground truth data for processing in network
            X_buff, y_buff = data_extraction_from_files(self.cases_folder, img_folder = ID)


            # extracts patch from data to process in network
            X[i,], y[i,] = crop_coords(self.dim, X_buff, y_buff, self.case_dims, ID)


            # introduces random augmentation on data if desired
            if self.augment== True:

                rand_ref_x = random.randint(0,1)
                rand_ref_y = random.randint(0,1)
                rand_ref_z = random.randint(0,1)
                rand_ref_rot_y = random.randint(0,1)

                if rand_ref_x == 0:
                    X[i,], y[i,] = X[i, :, ::-1], y[i, :, ::-1]

                if rand_ref_y == 0:
                    X[i,], y[i,] = X[i, :, :, ::-1], y[i, :, :, ::-1]

                if rand_ref_z == 0:
                    X[i,], y[i,] = X[i, :, :, :, ::-1], y[i, :, :, :, ::-1]

                if rand_ref_rot_y == 0:
                    rot_y = random.randint(0,4)
                    X[i,], y[i,] = np.rot90(X[i,], rot_y), np.rot90(y[i,], rot_y)


        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
