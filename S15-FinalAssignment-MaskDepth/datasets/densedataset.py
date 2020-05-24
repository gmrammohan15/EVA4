from os.path import splitext
import os
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2


class DenseDataSet(Dataset):
  def __init__(self, transform=None, scale=1):
    self.samples = []
    self.bg_dir = 'bgformat/bg'
    self.fg_bg_dir = 'dataset_training/fg_bg/'
    self.fg_bg_mask_dir = 'dataset_training/fg_bg_mask/'
    self.scale = 1
    self.transform = transform

    images = sorted([splitext(file)[0] for file in listdir(self.fg_bg_dir)
                if not file.startswith('.')])

    print(f'Creating dataset with {len(images)} examples')
    mask_images = sorted([splitext(file)[0] for file in listdir(self.fg_bg_mask_dir)
                if not file.startswith('.')])
    bg_paths = sorted(get_files(self.bg_dir))

    index = 0

    for i in range(len(bg_paths) - 90):
      bg_path = bg_paths[i]
      fg_bg_paths = sorted(glob(self.fg_bg_dir + images[index] + '/*.jpg'))
      mask_img_paths = sorted(glob(self.fg_bg_mask_dir + mask_images[index] + '/*.jpg'))
      count = 0
      for fg_bg_path in fg_bg_paths:
        self.samples.append((bg_path, fg_bg_path, mask_img_paths[count]))
        count = count + 1
      index = index + 1

  def __len__(self):
      return len(self.samples)

  @classmethod
  def preprocess(cls, pil_img, scale):
      w, h = pil_img.size
      newW, newH = int(scale * w), int(scale * h)
      assert newW > 0 and newH > 0, 'Scale is too small'
      pil_img = pil_img.resize((newW, newH))

      img_nd = np.array(pil_img)

      if len(img_nd.shape) == 2:
          img_nd = np.expand_dims(img_nd, axis=2)

      img_trans = img_nd.transpose((2, 0, 1))
      if img_trans.max() > 1:
          img_trans = img_trans / 255

      return img_trans

  def __getitem__(self, i):
      bg_path, fg_bg_path, fg_bg_mask_path = self.samples[i]

      # read fg, alpha
      bg_img = cv2.imread(bg_path)[:, :, :3]
      fgbg_img = cv2.imread(fg_bg_path)[:, :, :3]
      fgbg_mask_img = cv2.imread(fg_bg_mask_path)[:, :, 0]

      bg_img = cv2.resize(bg_img, (96, 96), interpolation=cv2.INTER_LINEAR)
      fgbg_img = cv2.resize(fgbg_img, (96, 96), interpolation=cv2.INTER_LINEAR)
      fgbg_mask_img = cv2.resize(fgbg_mask_img, (96, 96), interpolation=cv2.INTER_LINEAR)

      if self.transform:
        fgbg_img = self.transform(fgbg_img)
        bg_img = self.transform(bg_img)

      #return {'bg_image': bg_img, 'fgbg_image': fgbg_img,'mask': fgbg_mask_img}
      return {'fgbg_image': fgbg_img, 'bg_image': bg_img, 'mask': torch.from_numpy(fgbg_mask_img.astype(np.float32)[np.newaxis, :, :])}
