import tensorflow as tf
import numpy as np
import cv2
import os
import yaml
import time
import glob

from datasets.data_manager import DataManager
from datasets.augmentation import center_crop_3d, random_crop_3d
from datasets.tcia.tcia_pancreas_dataset import TCIAPancreasDataset
from datasets.tcia.tcia_patient import TCIAPatient

from prior import Prior

class TCIAPancreas3DDataManager(DataManager):

    def __init__(self,
                 data_dir,
                 split_name='split_1',
                 valid_split_number=0,
                 keep_train_annotations_p=1.0,
                 **_ignored):

        super().__init__(data_dir, split_name, valid_split_number)

        self.patient_type = TCIAPatient
        
        self.nb_classes = 2

        self.image_path = os.path.join(data_dir, 'volume_images')
        self.annotation_path = os.path.join(data_dir, 'volume_annotations')

        self.image_output_shape = TCIAPancreasDataset.VOLUME_SHAPE + (1,)

        assert keep_train_annotations_p >= 0.0 and keep_train_annotations_p <= 1.0
        self.keep_train_annotations_p = keep_train_annotations_p

        self.output_types = ({'input_1' : tf.float32}, tf.uint8)
        self.output_shapes = (
            {
                'input_1' : tf.TensorShape(self.image_output_shape)
            }, tf.TensorShape([self.image_output_shape[0], self.image_output_shape[1], self.image_output_shape[2], self.nb_classes])
        )



    def _read_split_file(self, filename):
        with open(filename, 'r') as f:
            line = f.read()
        patient_ids = [int(x) for x in line.split(';') if x != '']

        samples = []
        for pid in patient_ids:
            pid = str(pid).zfill(4)
            samples += ['{}.npy'.format(pid)]

        return samples


    def get_samples(self, mode):
        if mode == 'valid':
            annotations_file = os.path.join(self.data_dir, 'splits', '{}.{}'.format(self.split_name, self.valid_split_number))
            samples = self._read_split_file(annotations_file)
        elif mode == 'train':
            all_splits_files = glob.glob(os.path.join(self.data_dir, 'splits', '{}.*'.format(self.split_name)))
            all_splits_files = [x for x in all_splits_files if not x.endswith(str(self.valid_split_number))]
            print(all_splits_files)
            samples = []
            for split_file in all_splits_files:
                samples += self._read_split_file(split_file)
            # Remove pids is necessary : keep_train_annotations_p != 1.0
            samples = sorted(samples)
            samples = samples[0:round(len(samples)*self.keep_train_annotations_p)]
        else:
            raise Exception('Unknown mode {}'.format(mode))

        np.random.shuffle(samples)
        print(samples)
        return samples


    def _get_element(self, patient_id, mode, data_augmentation):
        filename = '{}.npy'.format(patient_id)
        image = np.load(os.path.join(self.image_path, filename))
        image = image.astype(np.float32)
        annotation = np.load(os.path.join(self.annotation_path, filename))

        d_w = self.image_output_shape[0] - image.shape[0]
        d_h = self.image_output_shape[1] - image.shape[1] 
        d_d = self.image_output_shape[2] - image.shape[2] 
            
        w_pad = d_w if d_w > 0 else 0
        h_pad = d_h if d_h > 0 else 0
        d_pad = d_d if d_d > 0 else 0

        image = np.pad(image, pad_width=((w_pad//2, w_pad//2+w_pad%2), (h_pad//2, h_pad//2+h_pad%2), (d_pad//2, d_pad//2+d_pad%2)), mode='edge')
        annotation = np.pad(annotation, pad_width=((w_pad//2, w_pad//2+w_pad%2), (h_pad//2, h_pad//2+h_pad%2), (d_pad//2, d_pad//2+d_pad%2)), mode='edge')

        transformation_info = {'theta' : 0,
                               'tx' : 0,
                               'ty' : 0,
                               'zx' : 0,
                               'zy' : 0,
                               'cx' : 0,
                               'cy' : 0,
                               'cz' : 0}
        if data_augmentation:
            theta = np.random.randint(-6, 6)
            transformation_info['theta'] = theta 
            tx = np.random.randint(-10, 10)
            transformation_info['tx'] = tx
            ty = np.random.randint(-10, 10)
            transformation_info['ty'] = ty
            zx = 1 + np.random.rand()*0.2-0.1
            transformation_info['zx'] = zx
            zy = 1 + np.random.rand()*0.2-0.1
            transformation_info['zy'] = zy

            image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy, fill_mode='reflect')
            annotation = tf.keras.preprocessing.image.apply_affine_transform(annotation, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy, fill_mode='reflect')

            sync_seed = np.random.randint(0, 4294967295)
            image, crop_info = random_crop_3d(image, self.image_output_shape, sync_seed=sync_seed)
            annotation, _ = random_crop_3d(annotation, self.image_output_shape, sync_seed=sync_seed)
            transformation_info['cx'] = crop_info[0]
            transformation_info['cy'] = crop_info[1]
            transformation_info['cz'] = crop_info[2]
        else:
            image, crop_info = center_crop_3d(image, self.image_output_shape)
            annotation, _ = center_crop_3d(annotation, self.image_output_shape)
            transformation_info['cx'] = crop_info[0]
            transformation_info['cy'] = crop_info[1]
            transformation_info['cz'] = crop_info[2]


        features = {}
        features['input_1'] = np.expand_dims(image, axis=-1)
        
        annotation = tf.keras.utils.to_categorical(annotation, num_classes=self.nb_classes)

        return (features, annotation), transformation_info
            
    
    def gen(self, examples_names, mode, data_augmentation=False):
        for i, filename in enumerate(examples_names):
            patient_id = filename.split('.')[0]
            element, _ = self._get_element(patient_id, mode, data_augmentation=data_augmentation)
            yield element





class TCIAPancreas3DPrior3DDataManager(TCIAPancreas3DDataManager):

    def __init__(self,
                 data_dir,
                 split_name='split_1',
                 valid_split_number=0,
                 keep_train_annotations_p=1.0,
                 prior_path=None,
                 train_info_filename=None,
                 valid_info_filename=None,
                 **_ignored):
        super(TCIAPancreas3DPrior3DDataManager, self).__init__(data_dir, split_name, valid_split_number, keep_train_annotations_p)

        if prior_path is None:
            prior_path = os.path.join(data_dir, 'priors', '3D_prior_3D_volume_{}_{}.npy'.format(split_name, valid_split_number))
        else:
            print(prior_path)
        self.prior = Prior(prior_path)

        self.train_info_filename_template = train_info_filename
        if self.train_info_filename_template is None:
            self.train_info_filename_template = '{}_Info_TCIA_pancreas_3D_position.yaml'
            
        self.valid_info_filename_template = valid_info_filename
        if self.valid_info_filename_template is None:
            self.valid_info_filename_template = '{}_Info_TCIA_pancreas_3D_position.yaml'

        print('Train position file info :', self.train_info_filename_template)
        print('Valid position file info :', self.valid_info_filename_template)
            
        self.output_types = ({'input_1' : tf.float32,
                              'prior_map' : tf.float32},
                             tf.uint8)
        self.output_shapes = (
            {
                'input_1' : tf.TensorShape(self.image_output_shape),
                'prior_map' : tf.TensorShape([self.image_output_shape[0],
                                           self.image_output_shape[1],
                                           self.image_output_shape[2],
                                           self.nb_classes])
            },
            tf.TensorShape([self.image_output_shape[0],
                            self.image_output_shape[1],
                            self.image_output_shape[2],
                            self.nb_classes])
            )


    def _get_element(self, patient_id, mode, data_augmentation):
        (features, annotation), transformation_info = super(TCIAPancreas3DPrior3DDataManager, self)._get_element(patient_id, mode, data_augmentation)

        # Get z_min and z_max for the pancreas annotation
        if mode == 'train':
            info_filename = self.train_info_filename_template
        else:
            info_filename = self.valid_info_filename_template

        with open(os.path.join(self.annotation_path, info_filename.format(patient_id)), 'r') as f:
            z_positions = yaml.load(f, Loader=yaml.FullLoader)

        # z_positions['1']['z_min'] = 54 - 62//2
        # z_positions['1']['z_max'] = 54 + 62//2

        if data_augmentation:
            z_offset = np.random.randint(-4, 4)
            z_positions['1']['z_min'] += z_offset
            z_positions['1']['z_max'] += z_offset
            
        prior_map = np.zeros_like(annotation)
        nb_slices = prior_map.shape[2]
        for s in range(nb_slices):
            prior_map[:,:,s,:] = self.prior.get_prior_for_slice(transformation_info['cz']+s, z_positions)

        features['prior_map'] = prior_map
        
        return (features, annotation), transformation_info
