from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():
  def __init__(self):
    self.train_transform = Compose([
      HorizontalFlip(),
      RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
      Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      Cutout(num_holes=3),
      ToTensor()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img
