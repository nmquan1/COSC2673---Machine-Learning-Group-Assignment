import os
import shutil
import hashlib
import matplotlib.pyplot as plt
from tabulate import tabulate

def compute_hash(file_path):
    with open(file_path, 'rb') as f:
        image_data = f.read()
        return hashlib.md5(image_data).hexdigest()

def find_duplicates(dataset_dir):
    hash_dict = {}
    duplicates = []
    count_duplicates = 0

    for root, dirs, files in os.walk(dataset_dir):
        for file in files: 
            file_path = os.path.join(root, file)
            file_hash = compute_hash(file_path)

            if file_hash in hash_dict:
                duplicates.append((file_path, hash_dict[file_hash]))
                count_duplicates += 1
            else:
                hash_dict[file_hash] = file_path

    return duplicates, count_duplicates

def print_duplicates(dataset_dir):
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
    if not isinstance(path, str):
        path = str(path)

    for root, dirs, _ in os.walk(path):
        depth = root.replace(path, '').count(os.sep)

        if depth == 0:
            indent = ''
        else:
            indent = ' ' * 4 * depth

        print(f"{indent}├───{os.path.basename(root)}/")  

def print_summary(path):
    data = []
    dir_name = []
    for dirpath, dirnames, filenames in os.walk(path):
        num_images = 0
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg')):
                num_images += 1

        if num_images > 0:
            dir_name.append(dirnames)
            relative_path = os.path.basename(dirpath)
            data.append([relative_path,num_images])
    headers = ['Directory', 'Number of Images']
    print(tabulate(data, headers=headers, tablefmt='pretty'))





    
