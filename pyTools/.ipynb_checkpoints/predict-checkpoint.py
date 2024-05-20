import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model


def vResults_tf(model: Model, dataset, class_labels: list):
    # Create a grid of subplots to visualize 16 samples
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()  # Flatten the array of axes
    
    # Get the first batch from the dataset
    first_batch = next(iter(dataset))  # Get the first batch
    x, y_true = first_batch  # Extract images and true labels
    x = np.array(x)
    
    # Predict on the first batch
    y_pred = model.predict(x, verbose=0)  # Predict using the model
    
    # Loop through the first 16 samples
    for i in range(min(16, len(x))):  # Ensure we don't exceed available samples
        # Get the true and predicted class indices
        y_pred_class = np.argmax(y_pred[i])  # Predicted class for the ith sample
        y_true_class = int(y_true[i])  # True class for the ith sample
        
        # Get the image (as-is, with no additional processing)
        image = x[i]  # Get the ith image
        
        # Display the image
        axes[i].imshow(image, cmap='gray')  # Assuming grayscale images
        axes[i].axis('off')  # No axis markings
        
        # Display predicted and true labels
        predicted_label = class_labels[y_pred_class]
        true_label = class_labels[y_true_class]
        
        # Determine if the prediction was correct
        is_correct = y_pred_class == y_true_class
        
        # Set the title and text color based on whether the prediction was correct
        color = 'green' if is_correct else 'red'
        axes[i].set_title(
            f'True: {true_label}\nPred: {predicted_label}',
            fontsize=8,
            color=color,
            fontweight='bold'
        )

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()



def vResults_raw(model: Model, raw_data, class_labels: list):
    # Create a grid of subplots to visualize 16 samples
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()  # Flatten the array of axes
    
    # If raw_data is not a list or iterable, raise an error
    if not isinstance(raw_data, (list, np.ndarray)):
        raise TypeError("raw_data should be a list or an array of images.")
    
  
    raw_data = np.array(raw_data)
    
    # Check if the shape of raw_data is compatible with the model
    if raw_data.shape[-1] != 1 or len(raw_data.shape) != 4:
        raise ValueError("raw_data should have shape (batch_size, 28, 28, 1).")
    
    # Predict on the raw data
    y_pred = model.predict(raw_data, verbose=0)  # Predict using the model
    
    # Loop through the first 16 samples
    for i in range(min(16, len(raw_data))):  # Ensure we don't exceed available samples
        # Get the predicted class index for the ith sample
        y_pred_class = np.argmax(y_pred[i])  # Predicted class
        
        # Get the image (as-is, with no additional processing)
        image = raw_data[i]
        
        # Display the image
        axes[i].imshow(image, cmap='gray')  # Grayscale display
        axes[i].axis('off')  # No axis markings
        
        # Display predicted label
        predicted_label = class_labels[y_pred_class]
        
        # Set the title with the predicted label
        axes[i].set_title(f'Pred: {predicted_label}', fontsize=8, fontweight='bold')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()


import random

def vResults_random(model: Model, dataset, class_labels: list, num_images: int = 16):
    # Create a grid of subplots to visualize the selected samples
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()  # Flatten the array of axes

    # Get all data and labels from the dataset
    all_images = []
    all_labels = []
    for batch in dataset:
        x, y = batch
        all_images.extend(x.numpy())
        all_labels.extend(y.numpy())
    
    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    # Randomly select indices for visualization
    selected_indices = random.sample(range(len(all_images)), num_images)
    selected_images = all_images[selected_indices]
    selected_labels = all_labels[selected_indices]

    # Predict on the selected images
    y_pred = model.predict(selected_images, verbose=0)  # Predict using the model

    # Loop through the selected samples
    for i in range(min(num_images, len(selected_images))):  # Ensure we don't exceed available samples
        # Get the predicted class index for the ith sample
        y_pred_class = np.argmax(y_pred[i])  # Predicted class

        # Get the image (as-is, with no additional processing)
        image = selected_images[i]
        
        # Display the image
        axes[i].imshow(image, cmap='gray')  # Grayscale display
        axes[i].axis('off')  # No axis markings

        # Display predicted and true labels
        predicted_label = class_labels[y_pred_class]
        true_label = class_labels[selected_labels[i]]

        # Set the title with the predicted label
        color = 'green' if y_pred_class == selected_labels[i] else 'red'
        axes[i].set_title(
            f'True: {true_label}\nPred: {predicted_label}',
            fontsize=8,
            color=color,
            fontweight='bold'
        )

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()
