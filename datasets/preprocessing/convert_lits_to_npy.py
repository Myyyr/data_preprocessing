
import os
import argparse
import numpy as np
import copy
import threading

import time


from datasets.lits.lits_dataset import LITSDataset

SAVE_MODES = ['slice', 'volume']

# Data array is saved as follow {target_directory}/pid/slice_num.npy
def serialize_slice(arr, patient_id, slice_num, target_directory, verbose=False):
    filename = os.path.join(target_directory, patient_id, '{}.npy'.format(slice_num))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if verbose:
        print('Saving file in : {}'.format(filename))
    np.save(filename, arr)

    
# Data array is saved as follow {target_directory}/pid.npy
def serialize_volume(arr, patient_id, target_directory, verbose=False):
    filename = os.path.join(target_directory, '{}.npy'.format(patient_id))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if verbose:
        print('Saving file in : {}'.format(filename))
    np.save(filename, arr)


processed_patient = 0
    
def browse_pids(pids, dataset, image_output_dir, annotation_output_dir, save_mode):
    global processed_patient
    
    for i, pid in enumerate(pids):
        s_time = time.time()
        image, annotation = dataset.get_patient_data_by_id(pid)
        if save_mode == 'slice':
            for s in range(image.shape[-1]):
                serialize_slice(image[:,:,s], pid, s, image_output_dir)
                serialize_slice(annotation[:,:,s], pid, s, annotation_output_dir)
        elif save_mode == 'volume':
            serialize_volume(image, pid, image_output_dir)
            serialize_volume(annotation, pid, annotation_output_dir)

        processed_patient += 1

        print('{} / {} ({:.2f}s)'.format(processed_patient, len(dataset), time.time()-s_time), end='\r')
    

def preprocess_and_serialize(data_dir, image_output_dir, annotation_output_dir, save_mode='slice', njobs=6):

    assert save_mode in SAVE_MODES

    if save_mode == 'slice':
        output_shape = LITSDataset.SLICE_SHAPE
    elif save_mode == 'volume':
        raise NotImplementedError()
    else:
        output_shape = (512, 512, -1)

    dataset = LITSDataset(data_dir, output_shape=output_shape)

    print("Start writing {} patients in ({}, {}), save_mode='{}'".format(len(dataset), image_output_dir, annotation_output_dir, save_mode))
    
    nb_patient = len(dataset.all_patients_ids)

    
    nb_patient_per_job = []
    for i in range(njobs+1):
        nb_patient_per_job.append(i * (nb_patient // njobs))
    nb_patient_per_job[-1] = nb_patient_per_job[-1] + nb_patient % njobs

    all_jobs = []
    for j in range(njobs):
        pids_to_browse = dataset.all_patients_ids[nb_patient_per_job[j]:nb_patient_per_job[j+1]]
        job = threading.Thread(target=browse_pids, args=(pids_to_browse, dataset, image_output_dir, annotation_output_dir, save_mode))
        job.start()
        all_jobs.append(job)

    for job in all_jobs:
        job.join()



if __name__ == "__main__":

    def argument_parser_def():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, required=True,
                            help='Path where the raw data is stored')
        parser.add_argument('--image_output_dir', type=str, required=True,
                            help='Path where the npy of the features will be written')
        parser.add_argument('--annotation_output_dir', type=str, required=True,
                            help='Path where the npy of the annotations will be written')
        parser.add_argument('--save_mode', type=str, default='slice',
                            help='Save mode could be "slice" or "volume"')
        return parser

    parser = argument_parser_def()
    FLAGS = parser.parse_args()

    preprocess_and_serialize(FLAGS.data_dir, FLAGS.image_output_dir, FLAGS.annotation_output_dir, save_mode=FLAGS.save_mode)
