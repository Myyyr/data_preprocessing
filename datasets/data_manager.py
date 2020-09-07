import tensorflow as tf
import numpy as np

import os




class DataManager:
    """ DataManager handle the input pipeline by reading the data and defining
    the input functions for training, evaluation and prediction.

    The data should be structured like this:
    DATA_DIR/annotations/<PID>/<SLICE_NUM>.npy : the annotation of slice SLICE_NUM from patient PID
    DATA_DIR/images/<PID>/<SLICE_NUM>.npy : the image of slice SLICE_NUM from patient PID

    Note that all data is stored as a serialised numpy array
    """

    def __init__(self,
                 data_dir,
                 split_name='split_1',
                 valid_split_number=0):

        self.output_types = None
        self.output_shapes = None
        
        self.data_dir = data_dir
        self.annotation_path = os.path.join(self.data_dir, 'annotations')
        self.image_path = os.path.join(self.data_dir, 'images')
        
        self.valid_split_number = valid_split_number
        self.split_name = split_name

        self.patient_type = None


    def get_samples(self, mode):
        raise NotImplementedError()

    
    def gen(self, examples_names, mode='train', data_augmentation=False):
        raise NotImplementedError()
    

    def dataset_by_examples(self, mode, examples_names, batch_size, nb_epochs, data_augmentation=False):
        ds = tf.data.Dataset.from_generator(lambda: self.gen(examples_names, mode, data_augmentation),
                                            self.output_types,
                                            self.output_shapes)
        ds = ds.repeat(nb_epochs)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(3)

        return ds

    
    def get_dataset(self, mode, batch_size, nb_epochs, data_augmentation=False):
        return self.dataset_by_examples(mode, self.get_samples(mode), batch_size, nb_epochs, data_augmentation=data_augmentation)


    def get_len(self, mode):
        examples = self.get_samples(mode)
        return len(examples)
