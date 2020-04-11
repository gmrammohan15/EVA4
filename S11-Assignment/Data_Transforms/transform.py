import torch
import torchvision
import torchvision.transforms as transforms
from .cutout import Cutout
from .train_transforms import TrainAlbumentation
from .test_transforms import TestAlbumentation

SEED = 1
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
#     transforms.RandomRotation((-10.0, 10.0)), transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
class Data():

  def __init__(self):
    self.train_album = TrainAlbumentation()
    self.test_album = TestAlbumentation()

  def getTrainDataSet(self, train=True):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.train_album)
    return dataset

  def getTestDataSet(self, train=False):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.test_album)
    return dataset

  def getTrainDataSetTorchTransforms(self, train=True):
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      transform_train.transforms.append(Cutout(n_holes=8, length=8))
      dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_train)
      return dataset     

  def getTestDataSetTorchTransforms(self, train=False):
      transform_test = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])
      dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_test)
      return dataset
      
  def getDataLoader(self, dataset, batches):
    # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batches, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader

    
  def getGradCamDataLoader(self, dataset):
  # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=1, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=1)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader
