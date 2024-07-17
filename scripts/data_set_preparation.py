import os
import shutil
import random

# Paths
dataset_dir = 'part3'
train_dir_male = 'data/train/male'
train_dir_female = 'data/train/female'
val_dir_male = 'data/validation/male'
val_dir_female = 'data/validation/female'

# Split ratio
train_split = 0.8

# Ensure directories exist
os.makedirs(train_dir_male, exist_ok=True)
os.makedirs(train_dir_female, exist_ok=True)
os.makedirs(val_dir_male, exist_ok=True)
os.makedirs(val_dir_female, exist_ok=True)

# Get all image filenames
all_images = os.listdir(dataset_dir)
random.shuffle(all_images)

# Split images into training and validation sets
train_images = all_images[:int(len(all_images) * train_split)]
val_images = all_images[int(len(all_images) * train_split):]

def move_images(images, src_dir, train_dst_male, train_dst_female, val_dst_male, val_dst_female):
    for img in images:
        gender = img.split('_')[1]
        if gender == '0':  # Male
            if img in train_images:
                shutil.move(os.path.join(src_dir, img), train_dst_male)
            else:
                shutil.move(os.path.join(src_dir, img), val_dst_male)
        elif gender == '1':  # Female
            if img in train_images:
                shutil.move(os.path.join(src_dir, img), train_dst_female)
            else:
                shutil.move(os.path.join(src_dir, img), val_dst_female)

# Move training images
move_images(train_images, dataset_dir, train_dir_male, train_dir_female, val_dir_male, val_dir_female)
# Move validation images
move_images(val_images, dataset_dir, train_dir_male, train_dir_female, val_dir_male, val_dir_female)