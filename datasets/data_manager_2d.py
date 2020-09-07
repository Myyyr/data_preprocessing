import tensorflow as tf
import numpy as np
import os
import math

from datasets.utils import get_slices_for_pids, get_pids_from_split_list
from utils import ProblemType

class DataManager2D:

    def __init__(self,
                 image_path,
                 annotation_path,
                 split_dir,
                 nb_classes,
                 batch_size=5,
                 split_name='split_1',
                 train_splits=[1,2,3,4],
                 valid_splits=[0],
                 test_splits=[],
                 problem_type=ProblemType.MULTICLASS,
                 add_ambiguity=False,
                 **_ignored):

        self.image_path = image_path
        self.annotation_path = annotation_path
        self.split_dir = split_dir

        self.split_name = split_name
        self.train_splits = train_splits
        self.valid_splits = valid_splits
        self.test_splits = test_splits

        self.batch_size = batch_size
        
        self.nb_classes = nb_classes
        self.add_ambiguity = add_ambiguity

        if isinstance(problem_type, str):
            self.problem_type = ProblemType.get_from_str(problem_type)
        else:
            self.problem_type = problem_type

        self.sequences = []
        self.NB_MAX_SEQUENCES = 5

    def set_annotation_path(self, new_annotation_path):
        self.annotation_path = new_annotation_path
        for s in self.sequences:
            s.annotation_path = new_annotation_path
        
    def get_sequence(self, mode, data_augmentation):
        if mode == 'valid':
            pids = get_pids_from_split_list(self.split_dir, self.split_name, self.valid_splits)
        elif mode == 'test':
            pids = get_pids_from_split_list(self.split_dir, self.split_name, self.test_splits)
        elif mode == 'train':
            pids = get_pids_from_split_list(self.split_dir, self.split_name, self.train_splits)
        else:
            raise Exception('Unknown mode {}'.format(mode))

        sequence = MedicalImageSliceSequence(pids,
                                             self.image_path,
                                             self.annotation_path,
                                             self.nb_classes,
                                             self.batch_size,
                                             data_augmentation,
                                             self.problem_type,
                                             self.add_ambiguity)
        
        if len(self.sequences) < self.NB_MAX_SEQUENCES:
            self.sequences.append(sequence)
        else:
            raise Exception('Too many sequences : {}'.format(len(self.sequences)))
        
        return sequence
        

        

class MedicalImageSliceSequence(tf.keras.utils.Sequence):

    def __init__(self,
                 pids,
                 image_path,
                 annotation_path,
                 nb_classes,
                 batch_size=5,
                 data_augmentation=True,
                 problem_type=ProblemType.MULTICLASS,
                 add_ambiguity=False,
                 shuffle=True):
        self.pids = pids
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.add_ambiguity = add_ambiguity

        if isinstance(problem_type, str):
            self.problem_type = ProblemType.get_from_str(problem_type)
        else:
            self.problem_type = problem_type

        self.samples = get_slices_for_pids(self.annotation_path, self.pids)
        if shuffle:
            np.random.shuffle(self.samples)

        
    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)
    

    def __getitem__(self, idx):
        if (idx + 1) * self.batch_size >= len(self.samples):
            batch_samples = self.samples[idx * self.batch_size:]
        else:
            batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = {'input_1' : []}
        if self.add_ambiguity:
            batch_x['ambiguity_map'] = []
        batch_y = []
        
        for s in batch_samples:
            patient_id, slice_num = s.split('.')[0].split('/')
            
            image = np.load(os.path.join(self.image_path, '{}/{}.npy'.format(patient_id, slice_num)))
            image = np.expand_dims(image, axis=-1)
            image = image.astype(np.float32)

            annotation = np.load(os.path.join(self.annotation_path, '{}/{}.npy'.format(patient_id, slice_num)))
        
            if self.add_ambiguity:
                ambiguity_map = np.array((annotation != -1), np.uint8)
                ambiguity_map = np.expand_dims(ambiguity_map, axis=-1)
                annotation[np.where(annotation == -1)] = 0

            annotation = np.expand_dims(annotation, axis=-1) # Needed to apply the affine transform

            if self.data_augmentation:
                theta = np.random.randint(-6, 6)
                tx = np.random.randint(-15, 15)
                ty = np.random.randint(-15, 15)
                zx = 1 + np.random.rand()*0.2-0.1
                zy = 1 + np.random.rand()*0.2-0.1

                image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy, fill_mode='nearest')
                annotation = tf.keras.preprocessing.image.apply_affine_transform(annotation, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy, fill_mode='nearest')
                if self.add_ambiguity:
                    ambiguity_map = tf.keras.preprocessing.image.apply_affine_transform(ambiguity_map, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy, fill_mode='nearest')

            if self.problem_type == ProblemType.MULTILABEL:
                annotation = tf.keras.utils.to_categorical(np.squeeze(annotation, axis=-1), num_classes=self.nb_classes+1)
                annotation = annotation[:,:,1:] # Ignore the background if in multilabels
            else:
                annotation = tf.keras.utils.to_categorical(np.squeeze(annotation, axis=-1), num_classes=self.nb_classes)

            batch_x['input_1'].append(image)
            if self.add_ambiguity:
                batch_x['ambiguity_map'].append(np.squeeze(ambiguity_map, axis=-1))

            batch_y.append(annotation)
            
        for k,v in batch_x.items():
            batch_x[k] = np.array(v)
            
        batch_y = np.array(batch_y)

        return batch_x, batch_y
