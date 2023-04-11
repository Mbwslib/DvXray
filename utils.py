import numpy as np
from PIL import Image
import torch
from scipy import ndimage

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def preprocess_input(image):
    image = np.array(image, dtype=np.float32)[:, :, ::-1]
    mean = [0.91584104, 0.9297611, 0.939562]
    std = [0.22090791, 0.1861283, 0.1651021]
    return (image / 255. - mean) / std

def adjust_learning_rate(optimizer, shrink_factor):

    print("\nDecaying learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def clip_gradient(optimizer, grad_clip):

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def confidence_weighted_view_fusion(ol_output, sd_output):

    total = torch.cat([torch.abs(ol_output - 0.5), torch.abs(sd_output - 0.5)], dim=0)
    coff = torch.softmax(total, dim=0)
    prediction = coff[0] * ol_output + coff[1] * sd_output

    return prediction

