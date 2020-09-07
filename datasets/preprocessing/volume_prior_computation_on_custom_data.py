import numpy as np
import glob
import os
import time
import yaml
from datasets import DataManagerFactory
from experimentation import NUMPY_RANDOM_SEED

from datasets.augmentation import center_crop_3d, pad_to_shape

import argparse


def compute_prior(nb_classes, frame_shape, data_dir, nb_slices_in_prior, verbose=0):

    organ_hist = np.zeros((nb_classes-1,
                           frame_shape[0],
                           frame_shape[1],
                           nb_classes,
                           nb_slices_in_prior))

    # pid.npy
    all_pids = set([x.split('.')[0] for x in os.listdir(data_dir) if x.endswith('.npy')])

    # Get region sizes
    regions_sizes = [0,]*nb_classes
    for k, pid in enumerate(all_pids):
        volume = np.load(os.path.join(data_dir, '{}.npy'.format(pid)))

        for class_id in range(1, nb_classes):
            class_annotation = (volume == class_id).astype(np.uint8)
            z_min, z_max = np.where(np.any(class_annotation, axis=(0,1)))[0][[0, -1]]
            size = z_max - z_min
            if regions_sizes[class_id] < size:
                regions_sizes[class_id] = size

    print("Regions sizes :", regions_sizes)
    regions_sizes[1] = 130
    print("Regions sizes :", regions_sizes)

    
    # Write z_min and z_max
    for k, pid in enumerate(all_pids):
        t1 = time.time()

        volume = np.load(os.path.join(data_dir, '{}.npy'.format(pid)))
        w, h, nb_slices = volume.shape
        
        full_annotation = volume

        zmin_zmax = {}
        for class_id in range(1, nb_classes):
            class_annotation = (full_annotation == class_id).astype(np.uint8)
            z_min, z_max = np.where(np.any(class_annotation, axis=(0,1)))[0][[0, -1]]
            z_mid = (z_min + z_max) // 2
            z_min = int(round(z_mid - regions_sizes[class_id] / 2))
            z_max = int(round(z_mid + regions_sizes[class_id] / 2))
            zmin_zmax[str(class_id)] = {
                        'z_min' : z_min,
                        'z_max' : z_max
                    }
            with open(os.path.join(data_dir, '{}_region_position.yaml'.format(pid)), 'w') as f:
                yaml.dump(zmin_zmax, f)
    
            for i in range(z_min, z_max, 1):
                idx_in_prior = (nb_slices_in_prior * (i-z_min)) // (z_max-z_min)
                for m in range(nb_classes):
                    organ_hist[class_id-1,:,:,m,idx_in_prior] += np.array(full_annotation[:,:,i] == m, dtype=np.uint8)

        if verbose==1:
            print('{} / {}  ({:.1f}%) {:.2f}s'.format(k+1, len(all_pids), (k+1)/len(all_pids)*100, time.time()-t1), end='\r')


    organ_prior = organ_hist / np.expand_dims(np.sum(organ_hist, axis=3), axis=3)
        
    return organ_prior

                  
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_classes', type=int,
                        help='Number of classes in the data')

    parser.add_argument('--frame_shape', type=int, nargs="+",
                        help='The shape of a single frame.')

    parser.add_argument('--data_dir', type=str,
                        help='Where the data is located')
                
    parser.add_argument('-s', '--slices_in_prior', type=int,
                        help='Number of slices in the prior')

    parser.add_argument('-o', '--output_filename', type=str,
                        help='The name of the npy file.')
                  
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help="Verbose level, 0 or 1.")
    
    FLAGS = vars(parser.parse_args())

    np.random.seed(NUMPY_RANDOM_SEED)

    # _ = compute_prior('valid',
    #                   FLAGS['dataset'],
    #                   FLAGS['data_dir'],
    #                   FLAGS['slices_in_prior'],
    #                   FLAGS['keep_train_annotations_p'],
    #                   FLAGS['verbose'])

    organ_prior = compute_prior(FLAGS['nb_classes'],
                                FLAGS['frame_shape'],
                                FLAGS['data_dir'],
                                FLAGS['slices_in_prior'],
                                FLAGS['verbose'])

    np.save(FLAGS['output_filename'], organ_prior)
                  
    
if __name__ == "__main__":
    main()
