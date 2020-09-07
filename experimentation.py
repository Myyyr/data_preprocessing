import tensorflow as tf
import os
import numpy as np
import pprint
import shutil
import glob

from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.models import load_model

from medpy.metric.binary import hd, assd, dc

from datasets import DataManagerFactory
# from models import ModelFactory
# from callbacks import SaveModel
from utils import read_config_file, temp_seed

from datasets.patient import Patient
from datasets.preprocessing.convert_dataset_to_npy import get_dataset

from config import default_parameters


TF_RANDOM_SEED = 983333813
NUMPY_RANDOM_SEED = 345152969




def load_experimentation_parameters(config_file_path):
    required_paramters = list(default_parameters.keys())
    parameters = default_parameters
    custom_parameters = read_config_file(config_file_path)

    for k, v in custom_parameters.items():
        parameters[k] = v

    for k in required_paramters:
        if parameters[k] is None:
            raise Exception('Parameter {} is None'.format(k))
        if parameters[k] == 'None':
            parameters[k] = None

    return parameters


def create_partially_labeled_annotations(fully_labeled_folder, dst_folder, pids, proportion, nb_classes):
    # Copy annotations if dst_folder doesn't exist
    if not os.path.exists(dst_folder):
        print('Copy dir {} in {}'.format(fully_labeled_folder, dst_folder))
        shutil.copytree(fully_labeled_folder, dst_folder)

        for i in range(1,nb_classes+1): # Remove organs
            with temp_seed(8*i+3): # Different shuffle depending on the class
                pids = sorted(pids)
                np.random.shuffle(pids)
                kept_pids = pids[0:round(len(pids)*proportion)]
            
            print('Class {}, Pids with annotation : {}'.format(i, kept_pids))
        
            for pid in pids:
                if pid not in kept_pids:
                    all_slices = glob.glob(os.path.join(dst_folder, str(pid), '*.npy'))
                    for s in all_slices:
                        annotation = np.load(s).astype(np.int8)
                        if -1 not in annotation:
                            annotation[np.where(annotation == 0)] = -1
                        annotation[np.where(annotation == i)] = -1
                        np.save(s, annotation)




class Experimentation():

    def __init__(self, config_filename, initial_weights=None, initial_model=None):
        #tf.enable_eager_execution()
        tf.keras.backend.clear_session()

        # Random seeds initialization
        print('Initializing the random with tf random seed : {} and numpy random seed : {}'.format(TF_RANDOM_SEED, NUMPY_RANDOM_SEED))
        tf.random.set_random_seed(TF_RANDOM_SEED)
        np.random.seed(NUMPY_RANDOM_SEED)

        # load params from the config file (yaml)
        self.params = load_experimentation_parameters(config_filename)

        # get data manager
        self.dataset_name = self.params['dataset']
        data_manager_factory = DataManagerFactory()
        data_manager_cls = data_manager_factory.get_data_manager(self.dataset_name, params=self.params)

        self.data_manager = data_manager_cls(image_path=self.params['image_path'],
                                             annotation_path=self.params['annotation_path'],
                                             split_dir=self.params['split_path'],
                                             nb_classes = self.params['nb_classes'],
                                             batch_size=self.params['train_batch_size'],
                                             split_name=self.params['split_name'],
                                             train_splits=self.params['train_splits'],
                                             valid_splits=self.params['valid_splits'],
                                             test_splits=self.params['test_splits'],
                                             problem_type=self.params['problem_type'],
                                             add_ambiguity=self.params['add_ambiguity'])

        self.train_dataset = self.data_manager.get_sequence('train', data_augmentation=self.params['data_augmentation'])
        self.valid_dataset = self.data_manager.get_sequence('valid', data_augmentation=False)

        self.train_steps_per_epoch = len(self.train_dataset)
        print('Train steps per epoch : ', self.train_steps_per_epoch)

        self.valid_steps_per_epoch = len(self.valid_dataset)
        print('Validation steps per epoch : ', self.valid_steps_per_epoch)

        # Create Partially labeled dataset if needed
        if self.params['annotation_proportion'] < 1.0:
            dst_folder = os.path.join(self.params['data_dir'], 'partial_annotations', 'p{}'.format(str(int(self.params['annotation_proportion']*100)).zfill(2)))
            create_partially_labeled_annotations(self.params['annotation_path'],
                                                 dst_folder,
                                                 self.train_dataset.pids,
                                                 self.params['annotation_proportion'],
                                                 self.data_manager.nb_classes)

            self.data_manager.set_annotation_path(dst_folder)

        # Learning rate scheduler
        decay_rate = (self.params['lr']/self.params['final_lr'] - 1) / self.params['nb_epochs']
        print('DECAY RATE:', decay_rate)
        self.params['lr'] = tf.keras.optimizers.schedules.InverseTimeDecay(self.params['lr'],
                                                                           decay_steps=self.train_steps_per_epoch,
                                                                           decay_rate=decay_rate)
        # Get the model
        self.initialize_model(initial_model, initial_weights)



    def initialize_model(self, initial_model=None, initial_weights=None):

        self.model_name = self.params['model_name']
        os.makedirs(self.params['model_dir'], exist_ok=True)

        self.model_dir_name = self.params['model_dir'].split('/')[-1]

        model_factory = ModelFactory()
        self.model = model_factory.get_model(self.model_name, params=self.params)

        if initial_model is not None:
            print('Loading initial model : ', initial_model)
            self.model.model = load_model(initial_model)
        elif initial_weights is not None:
            print('Loading initial weights : ', initial_weights)
            self.model.build(input_shape=self.params['input_shape'])
            self.model.load_weights(initial_weights)
        else:
            self.model.build(input_shape=self.params['input_shape'])

        pprint.pprint(self.params)

        
    def _save_code(self):
        src_folder = os.path.dirname(os.path.realpath(__file__))
        dst_folder = os.path.join(self.params['model_dir'], 'model_src/')
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.copytree(src_folder, dst_folder, ignore=shutil.ignore_patterns('.*'))



    def get_callbacks(self, model_save_sub_folder=None):
        callbacks = []

        save_folder = self.params['model_dir']
        if model_save_sub_folder is not None:
            save_folder = os.path.join(self.params['model_dir'], model_save_sub_folder)

        for cls_name in self.params['callbacks']:
            if cls_name == 'save_model':
                callbacks.append(
                    SaveModel(save_folder, save_best=True, monitor='val_dice_score', mode='max')
                )
            elif cls_name == 'csvlogger':
                callbacks.append(
                    CSVLogger(os.path.join(self.params['model_dir'], 'training.log'))
                )
            elif cls_name == 'tensorboard':
                callbacks.append(
                    TensorBoard(self.params['model_dir'])
                )
            else:
                print('Unknown callback name: ', cls_name)

        return callbacks

        
    def train(self):
        # Save code before training for future reproductibilty
        self._save_code()
        # Launch Training
        self.model.train(train_dataset=self.train_dataset,
                         epochs=self.params['nb_epochs'],
                         train_steps_per_epoch=self.train_steps_per_epoch,
                         callbacks=self.get_callbacks(),
                         validation_dataset=self.valid_dataset,
                         validation_steps_per_epoch=self.valid_steps_per_epoch)

        
    def predict(self, inputs):
        return self.model.model.predict(inputs, batch_size=self.params['valid_batch_size'])


    def evaluate(self, subset='valid', csv_saving_file=None):
        if subset == 'valid':
            print(self.model.evaluate(self.valid_dataset, csv_saving_file=csv_saving_file))
        elif subset == 'train':
            print(self.model.evaluate(self.train_dataset, csv_saving_file=csv_saving_file))
        else:
            raise Exception('Unknown subset:', subset)


    def evaluate_metrics(self, splits, data_config_file):

        data_parameters = read_config_file(data_config_file)
        dataset = get_dataset(data_parameters['dataset'])(self.params['data_dir'], data_parameters['class_info'])

        HD = []
        ASSD = []
        DSC = []
        TP = 0
        FP = 0
        FN = 0

        eval_pids = self.valid_dataset.get_pids_from_split_list(splits)

        for pid in eval_pids:

            p = dataset.get_patient_by_id(pid=pid,
                                          size=data_parameters['size'],
                                          voxel_spacing=data_parameters['voxel_spacing'],
                                          windowing=data_parameters['windowing'])
            voxel_spacing = p.get_spacing()

            data = []
            y_true = []
            for slice_name in os.listdir(os.path.join(self.params['image_path'], str(pid))):
                data.append(np.load(os.path.join(self.params['image_path'], str(pid), slice_name)))
                y_true.append(np.load(os.path.join(self.params['annotation_path'], str(pid), slice_name)))
            data = np.array(data)
            y_true = np.array(y_true)

            data = np.expand_dims(data, axis=-1)
            probas = self.predict(data)

            y_pred = np.argmax(probas, axis=-1)


            HD.append(hd(y_pred, y_true, voxel_spacing))
            ASSD.append(assd(y_pred, y_true, voxel_spacing))
            DSC.append(dc(y_pred, y_true))

            TP += np.sum(np.logical_and(y_pred == 1, y_true == 1))
            FP += np.sum(np.logical_and(y_pred == 1, y_true == 0))
            FN += np.sum(np.logical_and(y_pred == 0, y_true == 1))

            print('HD :', np.mean(HD))
            print('ASSD :', np.mean(ASSD))
            print('DSC :', np.mean(DSC))
            print('TP :', TP)
            print('FP :', FP)
            print('FN :', FN)



    def predict_patient(self, vtk_image_path, data_config_file, slice_view='axial'):

        data_parameters = read_config_file(data_config_file)

        p = Patient(vtk_image_path,
                    output_size=data_parameters['size'],
                    output_spacing=data_parameters['voxel_spacing'],
                    windowing=data_parameters['windowing'],
                    class_info=data_parameters['class_info'])

        print('Loading patient at :', vtk_image_path)

        data = p.get_image()

        print(data.shape)
        if slice_view == 'axial':
            data = np.moveaxis(data, 2, 0)
        elif slice_view == 'sagittal':
            data = np.moveaxis(data, 1, 0)
        print(data.shape)
        input_image = np.expand_dims(data, axis=-1)
        outputs = self.predict(input_image)
        
        return outputs
