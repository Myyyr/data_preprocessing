import numpy as np
import numbers

from scipy.ndimage.interpolation import zoom


# Find at : https://github.com/keras-team/keras/issues/3338
def center_crop_3d(x, center_crop_size, **kwargs):
    centerw, centerh, centerd = x.shape[0]//2, x.shape[1]//2, x.shape[2]//2
    halfw, halfh, halfd = center_crop_size[0]//2, center_crop_size[1]//2, center_crop_size[2]//2
    return x[centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh, centerd-halfd:centerd+halfd], (centerw-halfw, centerh-halfh, centerd-halfd)

def center_crop_2d(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh], (centerw-halfw, centerh-halfh)


# def random_crop_3d(x, random_crop_size, sync_seed=None, **kwargs):
#     np.random.seed(sync_seed)
#     w, h, d = x.shape[0], x.shape[1], x.shape[2]
#     rangew = w - random_crop_size[0]
#     rangeh = h - random_crop_size[1]
#     centerd = x.shape[2]//2
#     halfd = random_crop_size[2]//2
#     offsetw = 0 if rangew == 0 else np.random.randint(rangew)
#     offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
#     return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1], centerd-halfd:centerd+halfd], (offsetw, offseth, centerd-halfd)

def random_crop_3d(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h, d = x.shape[0], x.shape[1], x.shape[2]
    offsetw = (x.shape[0]//2 - random_crop_size[0]//2) + np.random.randint(-10, 10)
    offseth = (x.shape[1]//2 - random_crop_size[1]//2) + np.random.randint(-10, 10)
    offsetd = (x.shape[2]//2 - random_crop_size[2]//2) + np.random.randint(-5, 5)
    
    if offsetw+random_crop_size[0] > w:
        offsetw = w - random_crop_size[0]
    if offseth+random_crop_size[1] > h:
        offseth = h - random_crop_size[1]
    if offsetd+random_crop_size[2] > d:
        offsetd = d - random_crop_size[2]

    if offsetw < 0:
        offsetw = 0
    if offseth < 0:
        offseth = 0
    if offsetd < 0:
        offsetd = 0

        
    return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1], offsetd:offsetd+random_crop_size[2]], (offsetw, offseth, offsetd)



def pad_to_shape(x, output_shape, mode='edge'):
    dim = len(x.shape)
    assert(dim == len(output_shape))
    
    pad_width = ()
    for i in range(dim):
        d = output_shape[i] - x.shape[i]
        p = d if d > 0 else 0
        pad_width += ((p//2, p//2+p%2),)
        
    return np.pad(x, pad_width=pad_width, mode=mode)


def np_resample(img, current_spacing, target_spacing, bspline_order=3, mode='constant'):
        if isinstance(target_spacing, numbers.Number):
            target_spacing = [target_spacing] * img.ndim
        
        zoom_factors = [old / float(new) for new, old in zip(target_spacing, current_spacing)]
        # zoom image
        img = zoom(img, zoom_factors, order=bspline_order, mode=mode)
    
        return img
