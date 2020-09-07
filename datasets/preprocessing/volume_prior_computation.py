import numpy as np
import glob
import os
import time
import yaml
from datasets import DataManagerFactory
from experimentation import NUMPY_RANDOM_SEED

from datasets.augmentation import center_crop_3d, pad_to_shape

import argparse


def compute_prior(subset, dataset, data_dir, nb_slices_in_prior, p=1.0, verbose=0):

    factory = DataManagerFactory()
    params = {
        'data_dir' : data_dir,
        'keep_train_annotations_p' : p
    }
    data_manager = factory.get_data_manager(dataset, params)

    organ_hist = np.zeros((data_manager.nb_classes-1,
                           data_manager.image_output_shape[0],
                           data_manager.image_output_shape[1],
                           data_manager.nb_classes,
                           nb_slices_in_prior))

    # pid.npy
    all_pids = set([x.split('.')[0] for x in data_manager.get_samples(subset)])

    # Get region sizes
    organs_regions_sizes = [0,]*data_manager.nb_classes
    for k, pid in enumerate(all_pids):
        volume = np.load(os.path.join(data_manager.annotation_path, '{}.npy'.format(pid)))
        #volume = pad_to_shape(volume, data_manager.image_output_shape[0:3])
        #volume = center_crop_3d(volume, data_manager.image_output_shape)

        for class_id in range(1, data_manager.nb_classes):
            class_annotation = (volume == class_id).astype(np.uint8)
            z_min, z_max = np.where(np.any(class_annotation, axis=(0,1)))[0][[0, -1]]
            size = z_max - z_min
            if organs_regions_sizes[class_id] < size:
                organs_regions_sizes[class_id] = size

    print("Organs' regions sizes :", organs_regions_sizes)
    organs_regions_sizes[1] = 62
    print("Organs' regions sizes :", organs_regions_sizes)

    
    # Write z_min and z_max
    for k, pid in enumerate(all_pids):
        t1 = time.time()

        volume = np.load(os.path.join(data_manager.annotation_path, '{}.npy'.format(pid)))
        volume = pad_to_shape(volume, data_manager.image_output_shape[0:3])
        volume, crop_info = center_crop_3d(volume, data_manager.image_output_shape)
        w, h, nb_slices = volume.shape
        
        full_annotation = volume

        zmin_zmax = {}
        for class_id in range(1, data_manager.nb_classes):
            class_annotation = (full_annotation == class_id).astype(np.uint8)
            z_min, z_max = np.where(np.any(class_annotation, axis=(0,1)))[0][[0, -1]]
            z_mid = (z_min + z_max) // 2
            z_min = int(round(z_mid - organs_regions_sizes[class_id] / 2))
            z_max = int(round(z_mid + organs_regions_sizes[class_id] / 2))
            zmin_zmax[str(class_id)] = {
                        'z_min' : z_min+crop_info[2],
                        'z_max' : z_max+crop_info[2]
                    }
            with open(os.path.join(data_manager.annotation_path, '{}_Info_{}_region_position.yaml'.format(pid, dataset)), 'w') as f:
                yaml.dump(zmin_zmax, f)
    
            for i in range(z_min, z_max, 1):
                idx_in_prior = (nb_slices_in_prior * (i-z_min)) // (z_max-z_min)
                for m in range(data_manager.nb_classes):
                    organ_hist[class_id-1,:,:,m,idx_in_prior] += np.array(full_annotation[:,:,i] == m, dtype=np.uint8)

        if verbose==1:
            print('{} / {}  ({:.1f}%) {:.2f}s'.format(k+1, len(all_pids), (k+1)/len(all_pids)*100, time.time()-t1), end='\r')


    organ_prior = organ_hist / np.expand_dims(np.sum(organ_hist, axis=3), axis=3)
        
    return organ_prior

                  
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        help='Dataset name')

    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory.')
                
    parser.add_argument('-s', '--slices_in_prior', type=int,
                        help='Number of slices in the prior')

    parser.add_argument('-p', '--keep_train_annotations_p', type=float, default=1.0,
                        help='Number of slices in the prior')

    parser.add_argument('-o', '--output_filename', type=str,
                        help='The name of the npy file.')
                  
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help="Verbose level, 0 or 1.")
    
    FLAGS = vars(parser.parse_args())

    np.random.seed(NUMPY_RANDOM_SEED)

    _ = compute_prior('valid',
                      FLAGS['dataset'],
                      FLAGS['data_dir'],
                      FLAGS['slices_in_prior'],
                      FLAGS['keep_train_annotations_p'],
                      FLAGS['verbose'])

    organ_prior = compute_prior('train',
                                FLAGS['dataset'],
                                FLAGS['data_dir'],
                                FLAGS['slices_in_prior'],
                                FLAGS['keep_train_annotations_p'],
                                FLAGS['verbose'])

    np.save(FLAGS['output_filename'], organ_prior)
                  
    
if __name__ == "__main__":
    main()
