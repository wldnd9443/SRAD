import torch
from torch.nn import functional as F
from torchvision.io import read_image


def bicubic_sampling(tensor, scale_factor):
    """
        input: 4D tensor(including batch)
        output: tensor 
    """
    tensor = tensor.to(torch.float32)
    result = F.interpolate(tensor, scale_factor=scale_factor,mode="bicubic")
    result = result.to(torch.uint8)
    return result

def bilinear_sampling(tensor, scale_factor):
    """
        input: 4D tensor(including batch)
        output: tensor 
    """
    tensor = tensor.to(torch.float32)
    result = F.interpolate(tensor, scale_factor=scale_factor,mode="bilinear")
    result = result.to(torch.uint8)
    return result