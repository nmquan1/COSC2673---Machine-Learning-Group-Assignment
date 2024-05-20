import os
import shutil
import hashlib
import matplotlib.pyplot as plt
from tabulate import tabulate

# Functions to check duplicates, returning the MD5 hash value of the image file as a hexadecimal string
def compute_hash(file_path):
    # Compute the hash value of an image file
    with open(file_path, 'rb') as f:
        image_data = f.read()
        # Return the image's hash value (in hex) 
        return hashlib.md5(image_data).hexdigest()

def find_duplicates(dataset_dir):
    # Find duplicate images in a dataset
    hash_dict = {} # Dictionary to store hash values of analysed images
    duplicates = [] # List to store duplicates images' directories
    count_duplicates = 0 # Duplicate counter

    for root, dirs, files in os.walk(dataset_dir):
        # Generate a loop that loops through all files in a sub-folder
        for file in files: 
            file_path = os.path.join(root, file) # Get the file's path
            file_hash = compute_hash(file_path) # Compute the current image's hash value

            # If the current image's hash value is found in the dictionary of hash values that have been 
            # looped through, append the current image's directory into duplicates array and increase
            # counter by 1
            if file_hash in hash_dict:
                duplicates.append((file_path, hash_dict[file_hash]))
                count_duplicates += 1
            # If current image is not found (not a duplicate), add the image's path and its 
            # hash value into the dictionary and continue finding duplicates
            else:
                hash_dict[file_hash] = file_path

    return duplicates, count_duplicates

def print_duplicates(dataset_dir):
    # Access to the returned value of find_duplicates(dataset_dir) function
    duplicates, count_duplicates = find_duplicates(dataset_dir)
    print("Number of duplicates found: ", count_duplicates)
    print("Duplicates: ")
    count_line = 1
    for duplicate in duplicates:
        print(count_line, ". ", duplicate, "\n")
        count_line += 1

def plot_learning_curve(model):
    train_loss = model.history['loss']
    val_loss = model.history['val_loss']
    train_metric = model.history['sparse_categorical_accuracy']
    val_metric = model.history['val_sparse_categorical_accuracy']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))

    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-o', label='Training loss')
    plt.plot(epochs, val_loss, 'b-o', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metric, 'r-o', label='Training Accuracy')
    plt.plot(epochs, val_metric, 'b-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def print_directory_tree(path):
    # Ensure 'path' is a string for 'os.walk' to work properly
    if not isinstance(path, str):
        path = str(path)

    for root, dirs, _ in os.walk(path):
        # Determine the depth of the current directory
        depth = root.replace(path, '').count(os.sep)
        
        # No indentation for the first (root) directory
        if depth == 0:
            indent = ''
        else:
            indent = ' ' * 4 * depth  # Indentation for subdirectories
        
        # Print the directory name with or without indentation
        print(f"{indent}├───{os.path.basename(root)}/")  

def print_summary(path):
    data = []
    dir_name = []
    # Iterate over each subdirectory
    for dirpath, dirnames, filenames in os.walk(path):
        # Count the number of image files in the current subdirectory
        num_images = 0
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg')):
                num_images += 1

        # If any images are found, print a relative path from the base directory
        if num_images > 0:
            dir_name.append(dirnames)
            # Get the relative path from the base directory
            relative_path = os.path.basename(dirpath)
            data.append([relative_path,num_images])
    headers = ['Directory', 'Number of Images']
    print(tabulate(data, headers=headers, tablefmt='pretty'))





    
