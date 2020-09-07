import numpy as np
import glob
import os
import time
import yaml
from data_manager import DataManagerFactory

import argparse


def compute_prior(dataset, data_dir, random_seed, nb_slices_in_prior, verbose=0):

    factory = DataManagerFactory()
    params = {
        'data_dir' : data_dir,
        'random_seed' : random_seed
    }
    data_manager = factory.get_data_manager(dataset, params)

    counter = {}
    for i in range(data_manager.nb_classes):
        counter[i] = [0,] * nb_slices_in_prior
    mean_nb_slices = []
    organ_hist = np.zeros((data_manager.input_size[0], data_manager.input_size[1], data_manager.nb_classes, nb_slices_in_prior))

    all_pids = data_manager.train_ids

    for k, pid in enumerate(all_pids):
        t1 = time.time()
        all_slices = sorted(glob.glob(os.path.join(data_manager.annotation_path, pid, '*.npy')), key=lambda f: int(f.split('/')[-1].split('.npy')[0]))
        
        nb_slices = len(all_slices)
        annotations = []
        for i, slice_image in enumerate(all_slices):
            load_image = np.load(slice_image)
            if len(load_image.shape) > 2:
                load_image = np.argmax(load_image, axis=-1)
            annotations.append(load_image)
        full_annotation = np.stack(annotations, axis=-1)

        visited_z_for_bg = []
        zmin_zmax = {}
        for class_id in range(1, data_manager.nb_classes):
            class_annotation = (full_annotation == class_id).astype(np.uint8)
            bg_annotations = (full_annotation == 0).astype(np.uint8)
            z_min, z_max = np.where(np.any(class_annotation, axis=(0,1)))[0][[0, -1]]
            zmin_zmax[str(class_id)] = {
                        'z_min' : int(z_min),
                        'z_max' : int(z_max)
                    }

    
            for i in range(z_min, z_max, 1):
                idx_in_prior = (nb_slices_in_prior * (i-z_min)) // (z_max-z_min)
                organ_hist[:,:,class_id,idx_in_prior] += np.array(class_annotation[:,:,i], dtype=np.uint8)
                counter[class_id][idx_in_prior] += 1
                if i not in visited_z_for_bg:
                    organ_hist[:,:,0,idx_in_prior] += np.array(bg_annotations[:,:,i], dtype=np.uint8)
                    visited_z_for_bg.append(i)
                    counter[0][idx_in_prior] += 1
        with open(os.path.join(data_manager.annotation_path, pid, 'Info_{}_position.yaml'.format(dataset)), 'w') as f:
            yaml.dump(zmin_zmax, f)


        if verbose==1:
            print('{} / {}  ({:.1f}%) {:.2f}s'.format(k+1, len(all_pids), (k+1)/len(all_pids)*100, time.time()-t1), end='\r')

    organ_prior = np.zeros_like(organ_hist)
    for i in range(1, data_manager.nb_classes):
        for j in range(nb_slices_in_prior):
            organ_prior[:,:,i,j] = organ_hist[:,:,i,j] / counter[i][j]

    organ_prior[:,:,0,:] = 1-np.sum(organ_prior[:,:,1:,:], axis=-2)
        
    # for i in range(nb_slices_in_prior):
    #     organ_prior[:,:,:,i] = organ_hist[:,:,:,i] / np.expand_dims(np.sum(organ_hist[:,:,:,i], axis=-1), axis=-1)
    return organ_prior

                  
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        help='Dataset name')

    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory.')
            
    parser.add_argument('--random_seed', type=int,
                        help='Seed for data shuffle reproductivity')
    
    parser.add_argument('--slices_in_prior', type=int,
                        help='Number of slices in the prior')

    parser.add_argument('--output_filename', type=str,
                        help='The name of the npy file.')
                  
    parser.add_argument('--verbose', type=int, default=1,
                        help="Verbose level, 0 or 1.")
    
    FLAGS = vars(parser.parse_args())

    organ_prior = compute_prior(FLAGS['dataset'],
                                FLAGS['data_dir'],
                                FLAGS['random_seed'],
                                FLAGS['slices_in_prior'],
                                FLAGS['verbose'])
    np.save(FLAGS['output_filename'], organ_prior)
                  
    
if __name__ == "__main__":
    main()
