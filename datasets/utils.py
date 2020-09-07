import os
import numpy as np
import SimpleITK as sitk


def read_split_file(filename):
    with open(filename, 'r') as f:
        line = f.read()
    patient_ids = [int(x) for x in line.split(';') if x != '']
    return patient_ids


def get_slices_for_pids(data_path, pids):
    samples = []
    for pid in pids:
        pid = str(pid)
        samples += [pid+'/'+x for x in os.listdir(os.path.join(data_path, pid)) if x.endswith('npy')]
    return samples


def get_pids_from_split_list(split_dir, split_name, split_numbers):
    all_splits_files = []
    for s in split_numbers:
        all_splits_files.append(os.path.join(split_dir, '{}.{}'.format(split_name, s)))
    pids = []
    for split_file in all_splits_files:
        pids += read_split_file(split_file)
    return pids

    








def prior_post_processing(prior_map, gamma):
    """
    Prior Map post processing
    given prior map should be of size [H, W, K]
    """
    non_zero_indices = np.where(prior_map != 0)
    # eps = np.min(prior_map[non_zero_indices]) / 2
    eps = 1e-7
    prior_map[non_zero_indices] = prior_map[non_zero_indices] ** gamma
    prior_map = prior_map / np.expand_dims(np.sum(prior_map, axis=-1), axis=-1)
    prior_map[np.where(prior_map == 0)] = eps

    return prior_map














def center_crop_2d(x, center_crop_size):
    centerh, centerw = x.shape[0]//2, x.shape[1]//2
    halfh, halfw = center_crop_size[0]//2, center_crop_size[1]//2
    moduloh, modulow = center_crop_size[0]%2, center_crop_size[1]%2

    crop_h = [centerh-halfh, centerh+halfh+moduloh]
    crop_w = [centerw-halfw, centerw+halfw+modulow]

    if crop_h[0] < 0:
        crop_h[0] = 0
    if crop_h[1] > x.shape[0]:
        crop_h[1] = x.shape[0]

    if crop_w[0] < 0:
        crop_w[0] = 0
    if crop_w[1] > x.shape[1]:
        crop_w[1] = x.shape[1]

    return x[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]


def center_crop_3d(x, center_crop_size):
    x = center_crop_2d(x, center_crop_size[0:2])
    centerd = x.shape[2]//2
    halfd = center_crop_size[2]//2
    modulod = center_crop_size[2]%2

    crop_d = [centerd-halfd, centerd+halfd+modulod]

    if crop_d[0] < 0:
        crop_d[0] = 0
    if crop_d[1] > x.shape[2]:
        crop_d[1] = x.shape[2]

    return x[:,:,crop_d[0]:crop_d[1]]


def pad_to_shape(x, output_shape, mode='constant', constant_values=0):
    dim = len(x.shape)
    assert(dim == len(output_shape))

    pad_width = ()
    for i in range(dim):
        d = output_shape[i] - x.shape[i]
        p = d if d > 0 else 0
        pad_width += ((p//2, p//2+p%2),)

    return np.pad(x, pad_width=pad_width, mode=mode, constant_values=constant_values)


def resize_with_crop_or_pad(img, output_shape, pad_mode='constant', constant_values=0):
    # Change the -1 to the size of the img at axis i
    output_shape = list(output_shape)
    for i, v in enumerate(output_shape):
        if v < 0:
            output_shape[i] = img.shape[i]

    # Crop
    dim = len(img.shape)
    if dim == 2:
        img = center_crop_2d(img, output_shape)
    if dim == 3:
        img = center_crop_3d(img, output_shape)

    # Pad
    img = pad_to_shape(img, output_shape, mode=pad_mode, constant_values=constant_values)
    return img


def create_itk_image(data, infos):
    data = np.moveaxis(data, -1, 0)
    itk_image = sitk.GetImageFromArray(data.astype(np.uint8))
    itk_image.SetOrigin(infos['origin'])
    itk_image.SetDirection(infos['direction'])
    itk_image.SetSpacing(infos['spacing'])
    return itk_image
    


def read_image_with_sitk(image_path):
    if os.path.isfile(image_path):
        return sitk.ReadImage(image_path)
    elif os.path.isdir(image_path):
        return sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(image_path))
    else:
        raise Exception('Impossible to read: ', image_path)


def get_array_from_itk_image(itk_image):
    data = sitk.GetArrayFromImage(itk_image) # axis order is (z,y,x)
    data = np.moveaxis(data, 0, -1) # swap axis to get order (y,x,z)
    return data



def resample_itk_img(itk_image, out_spacing, is_label=False):

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # Change the -1 to the spacing of the img at axis i
    out_spacing = list(out_spacing)
    for i, v in enumerate(out_spacing):
        if v < 0:
            out_spacing[i] = original_spacing[i]

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # Remove the first and last two slices to avoid artifacts
    return resample.Execute(itk_image)#[:,:,2:-2]
