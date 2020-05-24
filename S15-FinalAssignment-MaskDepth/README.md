# Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. 

## Solution


# Google Drive Link for dataset:
https://drive.google.com/drive/folders/1WNMZTW67JD1ujh4UcVrxZHG33TkFtrBa?usp=sharing

# Approach 
### First targeting to solve the mask pred problem independently.
From the problem statement, it is clear that we need encoder-decoder model.Since its not typical classification problem\
We need to have prediction image that can be compared to ground truth mask image.So we need upsampling once we derive the basic features.

## Dataset 

We have 400 K fg bg images for train/test \
We have 100 bg images for train/test \
We have 400 K mask images corresponding to fgbg images

### Input image format for the model
Sample code: \
      imgs = batch['fgbg_image'] \
      bgimg = batch['bg_image'] \
      imgs = torch.cat((imgs, bgimg), dim=1) 

### Transformations applied:
Pytorch Color augmentation: \

brightness = (0.8, 1.2) \
contrast = (0.8, 1.2) \
saturation = (0.8, 1.2) \
hue = (-0.1, 0.1) \
torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

### Image resizing
Was getting out of RAM issues if we try to load the original size  i.e 200 x 200 even for for batch size of 128 \
Hence we need to resize to smaller resolution image so that we can train on collab \
Did image re-sizing to 64 x 64, 96 x 96 and checked.Works for both and hence trained using 96 x 96 image size \

code for return value from dataset class from which we will derive the loader \

{'fgbg_image': fgbg_img, 'bg_image': bg_img, 'mask': torch.from_numpy(fgbg_mask_img.astype(np.float32)[np.newaxis, :, :])}

### Train and validation loaders
To validate the model built, decided will pick out only 20, 0000 images from the corpus we have \
Using torch random split , split the train and test images.Saved the model, and loaded when new set of images were trained.

Ratio: 90 % train, 10 % test \

Code: \

dataset = DenseDataSet(transform=train_transforms, input_fgbg_path, input_bg_path, gt_mask_path) \
batch_size = 128 \
val_percent = 0.1
n_val = int(len(dataset) * val_percent) \
n_train = len(dataset) - n_val \
train, val = random_split(dataset, [n_train, n_val]) \

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) \
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True) \


## Visualization of input data fg, fg_bg, gt_mask, gt_depth

Sample Scene images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/bg_images_readme.png)

Sample fg bg images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/fg_bg_readme.png)

Sample fg bg mask images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/fg_bg_mask_readme.png)

Sample depth images for fg bg 

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/dd_model_output_readme.png)

After Dataset transformations:

## Input fg bg
![image](https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/datasets/images/input_fg_bg_transfotmed.png)

## input masks
![image](https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/datasets/images/input_gt_mask.png)


# Model for mask prediction
As stated above we need encoder-decoder or down and up convolution model to solve this problem \
Investigated available architecures 
1.VGG-16 \
2.VGG-19 \
3.UNet \
4.Resnet 18/50 \

This one more or less falls into category of semantic segmentation problem.\
Since we don't need to do object detection, our loss functions can be much simpler \

Thought process: \
VGG models are very heavy(~ 130 M params), so no plans to use this provided the constraints we have. \

Deecided to play around and use UNet architecture for this purpose. \

Model details:\
Model params:17,269,121 \
Params size (MB): 65.88 \
Estimated Total Size (MB): 301.75 \

https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/models/unet.py


## Loss function:
### Dice Coefficient:
The applications using UNet typically uses Dice coefficient  loss function.This would not solve our problem.\
Tried using this loss function.It was not able to detect the edges properly.However it will cover and mask the foreground object.Edges are not taken care here \

### Sigmoid + absolute difference
abs_diff = torch.abs(target - pred) \
loss = abs_diff.mean(1, True) \

### Sigmoid + BCE
Tested with \
crit = nn.BCEWithLogitsLoss() => Sigmoid + BCE loss \

Sigmoid as activation layer at the end followed by BCE worked well for this purpose. \

Below are the results captured in tensorboard: \

## Tensorboard ground truth for masks while training/eval

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/datasets/images/gt_masks_tensorboard.png)
## Tensorboard prediction for masks while training/eval

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/datasets/images/pred_mask_tensorboard.png)

# Debugging and working with Collab:
While working with Collab, it very essential we save the model after every run.\
Collab runtime disconnects quite frequently, may run into RAM/CUDA issues.To save time, \
its essential to save the model.PyTorch provides some of the api's

### Saving the model and loading 
torch.save(net.state_dict(), \
            dir_checkpoint + f'CP_epoch{epoch + 1}.pth') \
       
 net.load_state_dict( \
    torch.load("checkpoints/CP_epoch10.pth", map_location=device) \
 ) \
 
### Using Tensorboard to capture the metrics and images
Code: \

from torch.utils.tensorboard import SummaryWriter \

writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}') \

writer.add_scalar('Loss/train', loss.item(), global_step) \
writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step) \
writer.add_scalar('Loss/test', val_score, global_step) \
writer.add_images('masks/true', true_masks, global_step) \
writer.add_images('masks/pred', masks_pred, global_step) \
#### Batch processing time
Used python time.time() to capture the time taken for batch processing.

### Viewing tensorboard on collab using ngrok
Code: \

get_ipython().system_raw('./ngrok http 6006 &') \
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" \
!tensorboard --logdir runs \

### Optimzer:
Used standard SGD for this purpose. \
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

### Schedular:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=2) 

### Learning rate:
Try reducing learning rate after 10th epoch by ratio of 10 % \

    if epoch >= 10 : \
        lr = 0.01 * 0.1 \
        for param_group in param_groups: \
            param_group['lr'] = lr \
 Within 20 epochs was getting desired results.          



# Depth + Mask prediction Model

For solving this problem we need \
1)Common head to dervice features \
2)Seperate decoder for Depth prediction and Mask prediction which accepts features as input derived from 1 step

Encoder : \
Used standard torch based ResNet model.Tested with Resnet18
https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/models/resnet_encoder.py

Depth and Mask Decoder: \
https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/models/densedepth_decoder.py

Code: 

DepthMaskEncoder = models.ResnetEncoder(18, False) \
DepthMaskDecoder = models.DepthDecoder(num_channels, out_channels, scale) \

features = DepthMaskEncoder(torch.cat(inputs["fgbg_image"], inputs["bg_image"]) \
outputs = DepthMaskDecoder(features)
   
## Model params
Encoder params: 11,176,512 \
Decoder params :3152724 

## Depth masks prediction
![image](https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/datasets/images/depth_pred_tensorboard.png)

## Depth input 
![image](https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/datasets/images/depth_input_tensorboard.png)

## Loss function:
For Depth prediction , SSIM(Structural similarity) loss has been used \
However i could not find a common loss function that works for both Mask and Depth prediction \
Therefore, i could initialize the program for only purpose at given point in time.


## Collab and github links
https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/S15Final_resnetDepthMask.ipynb
https://github.com/gmrammohan15/EVA4/blob/master/S15-FinalAssignment-MaskDepth/S15Final_unet.ipynb
