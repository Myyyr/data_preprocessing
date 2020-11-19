import os
import numpy as np

from datasets.patient import Patient
from utils import read_config_file

class TCIAPancreasDataset():

    SLICE_SHAPE = (512,512)
    VOLUME_SHAPE = (144,144,96)
    #VOLUME_SHAPE = (160,160,96)
    
    def __init__(self, data_dir, class_info=None):
        self.data_dir = data_dir
        self.annotation_path = os.path.join(self.data_dir, 'annotations')
        self.image_path = os.path.join(self.data_dir, 'images')

        self.annotation_path_template = os.path.join(self.data_dir, 'annotations/label{}.nii')
        self.image_path_template = os.path.join(self.data_dir, 'images', 'PANCREAS_{}')
        
        self.all_patients_ids = [int(x.split('.')[0].split('label')[-1]) for x in os.listdir(self.annotation_path)]
        self.nb_classes = 2


    def get_patient_by_id(self, pid, size=(512,512,256), voxel_spacing=None, windowing=(-160,300)):
        annotation_path = self.annotation_path_template.format(str(pid).zfill(4))
        image_path = self.image_path_template.format(str(pid).zfill(4))
        
        p = Patient(image_path,
                    annotation_path=annotation_path,
                    output_size=size,
                    output_spacing=voxel_spacing,
                    windowing=windowing,
                    class_info=None,
                    name=pid)
        return p

        
    def get_patient_data_by_id(self, pid, size=(512,512,256), voxel_spacing=None, windowing=(-160,300)):
        p = self.get_patient_by_id(pid, size, voxel_spacing, windowing)
        return (p.get_image(), p.get_annotation())

    
    def __len__(self):
        return len(self.all_patients_ids)
        
