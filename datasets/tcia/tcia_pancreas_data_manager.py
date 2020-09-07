import tensorflow as tf
import numpy as np
import cv2
import os
import yaml
import time
import glob

from datasets.data_manager import DataManager

from datasets.tcia.tcia_pancreas_dataset import TCIAPancreasDataset

    
class TCIAPancreasDataManager(DataManager):

    def __init__(self,
                 data_dir,
                 split_name='split_1',
                 valid_split_number=0,
                 **_ignored):
        
        super().__init__(data_dir, split_name, valid_split_number)

        self.nb_classes = 2

        self.image_path = os.path.join(data_dir, 'images')
        self.annotation_path = os.path.join(data_dir, 'annotations')

        self.image_output_shape = TCIAPancreasDataset.SLICE_SHAPE
        
        # self.output_types = ({
        #     'image' : tf.float32,
        #     'patient_id' : tf.int32,
        #     'slice_num' : tf.int32,
        # }, tf.uint8)

        # self.output_shapes = ({
        #     'image' : tf.TensorShape(self.image_output_shape),
        #     'patient_id' : tf.TensorShape([]),
        #     'slice_num' : tf.TensorShape([]),
        # }, tf.TensorShape([self.image_output_shape[0], self.image_output_shape[1], self.nb_classes]))

        self.output_types = (tf.float32, tf.uint8)
        self.output_shapes = (tf.TensorShape(self.image_output_shape), tf.TensorShape([self.image_output_shape[0], self.image_output_shape[1], self.nb_classes]))
        
    def _read_split_file(self, filename):
        with open(filename, 'r') as f:
            line = f.read()
        patient_ids = [int(x) for x in line.split(';') if x != '']
        
        samples = []
        for pid in patient_ids:
            pid = str(pid).zfill(4)
            samples += [pid+'/'+x for x in os.listdir(os.path.join(self.annotation_path, pid))]
        
        return samples

        
    def get_samples(self, mode):
        if mode == 'valid':
            annotations_file = os.path.join(self.data_dir, 'splits', '{}.{}'.format(self.split_name, self.valid_split_number))
            samples = self._read_split_file(annotations_file)
        elif mode == 'train':
            all_splits_files = glob.glob(os.path.join(self.data_dir, 'splits', '{}.*'.format(self.split_name)))
            all_splits_files = [x for x in all_splits_files if not x.endswith(str(self.valid_split_number))]
            samples = []
            for split_file in all_splits_files:
                samples += self._read_split_file(split_file)
        else:
            raise Exception('Unknown mode {}'.format(mode))

        samples = sorted(samples)
        np.random.shuffle(samples)
        return samples

               
    def gen(self, examples_names, data_augmentation=False):
        
        for i, filename in enumerate(examples_names):
            patient_id, slice_num = filename.split('.')[0].split('/')
            
            image = np.load(os.path.join(self.image_path, filename))    
            image = np.expand_dims(image, axis=-1)
            image = image.astype(np.float32)
            
            annotation = np.load(os.path.join(self.annotation_path, filename))
            annotation = np.expand_dims(annotation, axis=-1) # Needed to apply the affine transform

            if data_augmentation:
                theta = np.random.randint(-6, 6)
                tx = np.random.randint(-30, 30)                                                                                                                                                            
                ty = np.random.randint(-30, 30)
                zx = 1 + np.random.rand()*0.2-0.1
                zy = 1 + np.random.rand()*0.2-0.1

                image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy, fill_mode='reflect')
                annotation = tf.keras.preprocessing.image.apply_affine_transform(annotation, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy, fill_mode='reflect')

            annotation = tf.keras.utils.to_categorical(np.squeeze(annotation, axis=-1), num_classes=self.nb_classes)
            
            features = {
                'image' : image,
                'patient_id' : int(patient_id),
                'slice_num' : int(slice_num),
            }
            yield (image, annotation)


