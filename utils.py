import tensorflow as tf
import os
import numpy as np
import itertools
import json
import yaml
import contextlib

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from enum import Enum

class ProblemType(Enum):
    MULTICLASS = 1
    MULTILABEL = 2

    def get_from_str(value):
        if value == 'multiclass':
            return ProblemType.MULTICLASS
        elif value == 'multilabel':
            return ProblemType.MULTILABEL
        else:
            raise Exception('Unknown problem type :', value)


def np_softmax(a):
    return np.exp(a) / (np.expand_dims(np.sum(np.exp(a), axis=-1), axis=-1) + 1e-7)

def np_sigmoid(a):
    return 1/(1 + np.exp(-a) + 1e-7) 


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def tf_multilabel_prediction(probas_pred, th=0.5):
    binary_pred = tf.cast(probas_pred > th, dtype=tf.int64)
    where_at_least_one_class_predicted = tf.cast(tf.reduce_sum(binary_pred, axis=-1) > 0, dtype=tf.int64)
    return (tf.argmax(probas_pred, axis=-1) + 1) * where_at_least_one_class_predicted

def np_multilabel_prediction(probas_pred, th=0.5):
    binary_pred = (probas_pred > th).astype(np.int8)
    where_at_least_one_class_predicted = (np.sum(binary_pred, axis=-1) > 0).astype(np.int8)
    return (np.argmax(probas_pred, axis=-1) + 1) * where_at_least_one_class_predicted 
    
        


def read_config_file(file_path):
    if not os.path.isfile(file_path):
        raise Exception('{} does not exists'.format(file_path))
    if file_path.split('.')[-1].lower() != 'yaml':
        raise Exception('{} is not a yaml file'.format(file_path))
    with open(file_path) as f:
        file_content = yaml.load(f, Loader=yaml.FullLoader)
    return file_content


def get_class_info(json_path):
    with open(json_path, 'r') as f:
        info = json.load(f)
    return info



def create_dir_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)



def get_2D_bbox(img, margin=0):                            
    assert len(img.shape) == 2                                        
    ax1 = np.where(np.any(img, axis=(0)))[0] 
    ax0 = np.where(np.any(img, axis=(1)))[0]

    ax0_min_max = (ax0.min()-margin, ax0.max()+1+margin)
    ax1_min_max = (ax1.min()-margin, ax1.max()+1+margin)    

    return (ax0_min_max, ax1_min_max)


def save_figure(original_image, predicted_image, ground_truth, title, o_filename, BB=None):
        
    fig = plt.figure(figsize=(10, 6))
    plt.title(title)
    
    cmap = mpl.colors.ListedColormap(['whitesmoke', 'dodgerblue'])
    bounds = np.linspace(0,2,3)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax1 = fig.add_subplot(1, 3, 1)
    mat1 = ax1.imshow(original_image, cmap='gray', norm=None)
    bar1 = plt.colorbar(mat1, orientation='horizontal')
    plt.title('Original')

    ax2 = fig.add_subplot(1, 3, 2)
    mat2 = ax2.matshow(ground_truth, cmap=cmap, norm=norm)
    if BB is not None:
        rect = patches.Rectangle((BB[2],BB[0]),BB[3]-BB[2],BB[1]-BB[0],linewidth=1,edgecolor='r',facecolor='none')                                                                                      
        ax2.add_patch(rect)
    bar2 = plt.colorbar(mat2, orientation='horizontal')
    ax2.xaxis.set_ticks_position('bottom')
    plt.title('Ground Truth')

    ax3 = fig.add_subplot(1, 3, 3)
    mat3 = ax3.matshow(predicted_image, cmap=cmap, norm=norm)
    if BB is not None:
        rect = patches.Rectangle((BB[2],BB[0]),BB[3]-BB[2],BB[1]-BB[0],linewidth=1,edgecolor='r',facecolor='none')
        ax3.add_patch(rect)
    bar3 = plt.colorbar(mat3, orientation='horizontal')
    ax3.xaxis.set_ticks_position('bottom')
    plt.title('Prediction')

    # ax4 = fig.add_subplot(1, 4, 4)
    # mat4 = ax4.matshow(prior_map, cmap=cmap, norm=norm)
    # bar4 = plt.colorbar(mat4, orientation='horizontal')
    # ax4.xaxis.set_ticks_position('bottom')
    # plt.title('Prior Map')

    
    plt.savefig(o_filename)
    plt.close()


def one_hot(array, nb_classes):
    height, width = array.shape
    one_hot_annotation = np.zeros((height, width, nb_classes), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            one_hot_annotation[i, j, array[i,j]] = 1
    return one_hot_annotation


def read_jsonfile(json_filename):
    with open(json_filename, 'r') as f:
        infos = f.read()
    infos = json.loads(infos)
    return infos




def device_selection(devices_argument):
    for device in devices_argument:
        assert ((device >= '0') and (device <= '3') or device == ',')
    os.environ["CUDA_VISIBLE_DEVICES"] = devices_argument
    print(os.environ["CUDA_VISIBLE_DEVICES"])


def bbox2_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def _masks_assembly(masks):
    # Assemble all masks in input masks which should be a list of numpy array
    # and return a numpy array with indexes as values
    assert len(masks) > 0
    res = np.zeros_like(masks[0], dtype=np.uint8)
    for i, mask in enumerate(masks):
        mask = np.array((mask / 255) * (i+1), dtype=np.uint8)
        res = res + mask
        # superpositions
        res[np.where(res > (i + 1))] = (i+1)
    return res


def crop_center_along_axis(image, desired_axis_length, axis=0):
    assert axis in [0, 1]
    axis_length = image.shape[axis]
    start_pos = int((axis_length / 2) - (desired_axis_length / 2))
    end_pos = int((axis_length / 2) + (desired_axis_length / 2))
    if axis == 0:
        return image[start_pos:end_pos,:]
    if axis == 1:
        return image[:,start_pos:end_pos]



def numpy_resize_image_with_crop_or_pad(image, target_h, target_w):
    resized_image = copy.deepcopy(image)
    origial_h, original_w, _ = image.shape

    if origial_h > target_h:
        resized_image = crop_center_along_axis(resized_image, target_h, axis=1)
    if original_w > target_w:
        resized_image = crop_center_along_axis(resized_image, target_w, axis=0)
    return resized_image
