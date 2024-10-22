from functools import partial
from keras import backend as K

# smoothing factor to avoid division by zero
smoothing_factor = 1e-6

# common version for computation of dice coefficient over classes
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# loss version of dice coefficient
def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


# function that computes weighted dice coefficient for tissue segmentation
def weighted_dice_coefficient(y_true, y_pred, axis=(0, 1, 2, 3), smooth=1e-5):
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))
# loss version of weighted dice coefficient
def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


# Function to evaluate specific tissue (class) dice index
class tissue_dice(object):
    def __init__(self, class_id, tissue_name):
        self.class_id = class_id
        self.__name__ = tissue_name

    # returns calculated tissue dice when called
    def __call__(self, y_true, y_pred):
        return self.tissue_dice(y_true, y_pred)

    # calculates tissue dice in Keras
    def tissue_dice(self, y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # calculates dice from true and predicted labels
        y_true_f = K.cast(K.equal(class_id_true, self.class_id), 'float32')
        y_pred_f = K.cast(K.equal(class_id_preds, self.class_id), 'float32')
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)


# Function to evaluate specific tissue (class) accuracy index
class tissue_accuracy(object):
    def __init__(self, class_id, tissue_name):
        self.class_id = class_id
        self.__name__= tissue_name

    # returns calculated tissue accuracy when called
    def __call__(self, y_true, y_pred):
        return self.tissue_calc( y_true, y_pred)

    # calculates tissue accuracy in Keras
    def tissue_calc(self, y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)

        accuracy_mask = K.cast(K.equal(class_id_preds, self.class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32')*accuracy_mask
        class_acc = K.sum(class_acc_tensor)/K.maximum(K.sum(accuracy_mask),1)
        return class_acc


# function to obtain learning rate at which model training is being executed
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# evaluates model recall on training and validation data
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# evaluates model precision on training and validation data
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true *y_pred,0,1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
