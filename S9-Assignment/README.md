#Assignment 9

ResNet18 model used from https://github.com/kuangliu/pytorch-cifar 

Used default ResNet18 code 

Used pytorch albumentations for data augmentation.Used as a module

Used our data loader, model loading, train, and test code to train ResNet18 on Cifar10

Used GradCam as module to show heatmap

Target test accuracy 87 %


#Assignment details:

Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)

Please make sure that your test_transforms are simple and only using ToTensor and Normalize

Implement GradCam function as a module. 

Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality

Target Accuracy is 87%

