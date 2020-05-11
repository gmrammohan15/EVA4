# Preparing Custom Data set for Background Subtraction and Depth Estimation
________
Data set for background subtraction shoulds have traning images of a object and a background where as target will have only the object Data set for depth prediction should have traning images of a object and a background where as target will have the depth map for the same.

# Details around requirement:
You must have 100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images. 

In total you MUST have: 

400k fg_bg images\
400k depth images\
400k mask images\
generated from:\
100 backgrounds\
100 foregrounds, plus their flips\
20 random placement on each background.\
Now add a readme file on GitHub for Project 15A:\
Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file. \
Add your dataset statistics:\
Kinds of images (fg, bg, fg_bg, masks, depth)\
Total images of each kind\
The total size of the dataset\
Mean/STD values for your fg_bg, masks and depth images\
Show your dataset the way I have shown above in this readme\
Explain how you created your dataset\
how were fg created with transparency\
how were masks created for fgs\
how did you overlay the fg over bg and created 20 variants\
how did you create your depth images? \
Add the notebook file to your repo, one which you used to create this dataset\
Add the notebook file to your repo, one which you used to calculate statistics for this dataset\



# Google Drive Link:
https://drive.google.com/drive/folders/1WNMZTW67JD1ujh4UcVrxZHG33TkFtrBa?usp=sharing


# Dataset statistics
### fg transparent background images: 
Total number of fg images: 100\
size: 1 MB\
Classes: dog, cat, humans\

### fg mask images: 
Total number of fg mask images: 100\
size: 400 kb\
Classes: dog, cat, humans\

### bg images:
Total number of bg mask images: 100\
size: 852 kb\
Classes: river, street, beach\

### fg bg images:
total images: 400 K
size: 1.8G (zipped)

### fg bg mask images
total images: 400K

### depth images of fg_bg
total images: 100 K \

# Visualization

Sample Scene images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/bg_images_readme.png)

Sample fg images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/fg_transparent_readme.png)

Sample fg mask images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/fg_mask_readme.png)

Sample fg bg images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/fg_bg_readme.png)

Sample fg bg mask images

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/fg_bg_mask_readme.png)

Sample depth images for fg bg 

![image](https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/dd_model_output_readme.png)


# Explanation of how data set is created

### Background images
Downloaded stanford data set for "scene" images.Total Zip was around 2GB.
Randomly selected 100 images.These are treated as bg images for our purpose
Use Python imaging library(PIL) for resizing the images to square images

### Foreground images
Randomly selected images for people, dogs, cats.We have to remove the back ground from this image. We have used Microsoft Powerpoint to remove the background, or any tools similar to photoshop can be used to do the same.
Use Python imaging library(PIL) for resizing the images to square images of size 200x200

### Mask for fg images
Used python cv2 lib to mask the foreground images


### overlay the fg over bg and create 20 variants
We have 100 bg images and 100 fg images
For each bg image , fg image is pasted randomly for 10 times and flipped 10 times.
So it would be 100 x 100 x 20(random + flip)  = 400K images

Basically, \
1.Have a Top FOR loop to run over the bg images\
2.Have one more FOR loop inside the above for loop to run over fg images\
3.Inside the second for loop, iterate for 20 times for flip and random overlay of fg images over the bg image at that point\

For the 400 K images above, generate mask.We use same method for creating fg masks here.

### Depth images
Used the below repo and pretrained model to predict the depth of fg bg images\
https://github.com/ialhashim/DenseDepth
Since the number of images is very huge, we need increase the RAM allocation in collab to 25 gb which comes free.\
Run the prediction over batch of images , instead runnning all at once\
Keep saving the prediction output into ZIP output.Python has good api's to work with ZIP files

# Repo links
Data set creation\
https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/create.ipynb

Depth Model\
https://github.com/gmrammohan15/EVA4/blob/master/S15A-Assignment/DenseDepthModel.ipynb

