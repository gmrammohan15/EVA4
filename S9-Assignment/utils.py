import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def visualize_cam(mask, img):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    print(type(heatmap))
    result = heatmap+img
    result = result.div(result.max()).squeeze()
    
    return heatmap, result