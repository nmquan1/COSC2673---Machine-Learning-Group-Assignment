import os
import shutil
import tempfile
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Get the image dataset from the temporary dataset
img_height = 28
img_width = 28
batch_size = 100

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def convert_to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

# Function to get only leaf directories
def get_leaf_directories(data_dir):
    leaf_directories = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if not dirnames:
            leaf_directories.append(dirpath)
    return leaf_directories

def data_from_dir(data_dir):
    data = image_dataset_from_directory(
        data_dir,
        seed=42,
        image_size = (img_height, img_width),
        color_mode = "grayscale",
        batch_size = 100
    )
    return data
	
def preprocess28(image_path):
    img = Image.open(image_path)

    img = img.resize((28, 28))

    img = img.convert('L')

    img_array = np.array(img) 

    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def count_images(directory):
    return sum(1 for filename in os.listdir(directory) if filename.endswith('.png'))

def augment_images(directory, augmentations_needed):
    for filename in os.listdir(directory):
        if filename.endswith('.png') and not filename.startswith('aug'):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            for _ in datagen.flow(x, batch_size=1,
                                      save_to_dir=directory,
                                      save_prefix='aug',
                                      save_format='png'):
                augmentations_needed -= 1
                if augmentations_needed <= 0:
                    break

def calibrate_shape_dataset(directory):
    for main_dir in os.listdir(directory):
        main_path = os.path.join(directory, main_dir)
        if os.path.isdir(main_path):
            # Initialize a flag to check if any images are moved
            for subdir in os.listdir(main_path):
                subdir_path = os.path.join(main_path, subdir)
                print(f"Moving images from {subdir_path} to {main_path}")
                if os.path.isdir(subdir_path):
                    for filename in os.listdir(subdir_path):
                        if filename.endswith('.png'):
                            src = os.path.join(subdir_path, filename)
                            dst = os.path.join(main_path, filename)
                            shutil.move(src, dst)
                    shutil.rmtree(subdir_path)
                    print(f"Deleting directory {subdir_path}")
    target_count = 5000

    for main_dir in os.listdir(directory):
        main_path = os.path.join(directory, main_dir)
        if os.path.isdir(main_path):
            original_count = count_images(main_path)
            augmentations_needed = target_count - original_count
            if augmentations_needed > 0:
                print(f"Augmenting {augmentations_needed} images in directory {main_path}")
                augment_images(main_path, augmentations_needed)

    min_images = min(count_images(os.path.join(directory, subdir)) for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir)))

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            num_images = count_images(subdir_path)
            if num_images > min_images:
                files_to_remove = num_images - min_images
                for filename in os.listdir(subdir_path):
                    if files_to_remove <= 0:
                        break
                    if filename.endswith('.png'):
                        os.remove(os.path.join(subdir_path, filename))
                        files_to_remove -= 1
    print("Directory reorganization and image augmentation completed.")

def calibrate_type_dataset(directory):
    for main_dir in os.listdir(directory):
        main_path = os.path.join(directory, main_dir)
        if os.path.isdir(main_path):
            for sub_dir in os.listdir(main_path):
                sub_path = os.path.join(main_path, sub_dir)
                if os.path.isdir(sub_path):
                    new_path = os.path.join(directory, sub_dir)
                    print(f"Moving {sub_path} to {new_path}")
                    shutil.move(sub_path, new_path)
            print(f"Deleting directory {main_path}")
            shutil.rmtree(main_path)

    image_counts = {leaf_dir: count_images(os.path.join(directory, leaf_dir)) for leaf_dir in os.listdir(directory) if os.path.isdir(os.path.join(directory,leaf_dir))}
    print("Image counts per directory:", image_counts)
    
    if image_counts:
        max_images = max(image_counts.values())
    else:
        max_images = 0
        print("No images found in any directory.")
    
    for leaf_dir, count in image_counts.items():
        leaf_path = os.path.join(directory, leaf_dir)
        augmentations_needed = max_images - count
        if augmentations_needed > 0:
            print(f"Augmenting {augmentations_needed} images in directory {leaf_path}")
            augment_images(leaf_path, augmentations_needed)
    
    print("Directory reorganization and image augmentation completed.")