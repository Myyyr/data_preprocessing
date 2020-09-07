import os
import numpy as np

from datasets.utils import resample_itk_img, read_image_with_sitk, get_array_from_itk_image, resize_with_crop_or_pad, create_itk_image

class Patient():

    def __init__(self,
                 image_path,
                 annotation_path=None,
                 output_size=None,
                 output_spacing=None,
                 windowing=None,
                 class_info=None,
                 name=None):

        self.image_path = image_path
        self.annotation_path = annotation_path

        self.output_size = output_size
        self.output_spacing = output_spacing
        self.windowing = windowing
        self.class_info = class_info

        self.name = name

        # Read image
        self.itk_image = read_image_with_sitk(self.image_path)
        self.original_infos = {'size' : self.itk_image.GetSize(),
                               'spacing' : self.itk_image.GetSpacing(),
                               'origin' : self.itk_image.GetOrigin(),
                               'direction' : self.itk_image.GetDirection()}
        
        # Resample to given spacing
        if self.output_spacing is not None:
            self.itk_image = resample_itk_img(self.itk_image, self.output_spacing, is_label=False)

        # Read annotation
        if self.annotation_path is not None:
            self.itk_annotation = read_image_with_sitk(self.annotation_path)
            if self.output_spacing is not None:
                self.itk_annotation = resample_itk_img(self.itk_annotation, self.output_spacing, is_label=True)



    def _get_data(self, itk_img):
        # Get the numpy array
        data = get_array_from_itk_image(itk_img)

        # Change image reading direction by following the image Direction matrix
        direction = np.reshape(itk_img.GetDirection(), (3,3))
        for i, a in enumerate([1, 0, 2]):
            if np.diag(direction)[i] == -1:
                data = np.flip(data, axis=a)

        if self.output_size is not None:
            data = resize_with_crop_or_pad(data, self.output_size, pad_mode='constant', constant_values=np.min(data))

        return data

                
    def get_image(self):
        image_data = self._get_data(self.itk_image)

        # Image windowing : clipping
        if self.windowing is not None:
            image_data = np.clip(image_data, self.windowing[0], self.windowing[1])

        # Normalization
        image_mean = np.mean(image_data)
        image_std = np.std(image_data)
        image_data = (image_data - image_mean) / image_std
        
        return image_data.astype(np.float32)

    
    def get_annotation(self):
        if self.annotation_path is None:
            return None
        else:
            annot_data = self._get_data(self.itk_annotation)

            # Map the class idx in a linear range of values
            if self.class_info is not None:
                annot_unique = np.unique(annot_data)
                mapped_annot_data = np.zeros_like(annot_data, dtype=np.uint8)
                for c in range(len(self.class_info)):
                    if self.class_info[c]['label'] in annot_unique:
                        mapped_annot_data[np.where(annot_data == self.class_info[c]['label'])] = c+1
                    else:
                        print('[{}] WARNING: organs {} not in the annotation'.format(self.annotation_path, self.class_info[c]['organs']))
                annot_data = mapped_annot_data
                    
        return annot_data.astype(np.int8)


    def get_spacing(self):
        return self.itk_image.GetSpacing()

    def get_size(self):
        return self.get_annotation().shape


    def reverse_processing_for_prediction(self, pred):
        if self.class_info is not None:
            mapped_data = np.zeros_like(pred, dtype=np.uint8)
            for c in range(1, len(self.class_info)):
                mapped_data[np.where(pred == c)] = self.class_info[c]['label']
        pred = mapped_data
        
        pred = resize_with_crop_or_pad(pred, (self.original_infos['size'][0:2])+(-1.0,))
        infos = {
            'origin' : self.original_infos['origin'],
            'direction' : self.original_infos['direction'],
            'spacing' : self.itk_image.GetSpacing()
        }
        
        itk_pred = create_itk_image(pred, infos)
        itk_pred = resample_itk_img(itk_pred,
                                    out_spacing=self.original_infos['spacing'],
                                    is_label=True)
        print(itk_pred.GetSpacing())

        return itk_pred
        
        
