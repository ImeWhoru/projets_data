import cv2
import numpy as np
import os
from tqdm import tqdm

image_folder = 'project_code/assets/img_celeba/'
output_path = 'average_image.jpg'

def compute_average_image(image_folder, output_path, resize_dim=None):
    """
    Compute the average image from all images in a folder.
    
    Parameters:
        image_folder (str): Path to the folder containing images.
        output_path (str): Path to save the resulting average image.
        resize_dim (tuple): Dimensions to resize all images (width, height). If None, uses original size.
    """
    # Initialize variables
    total_images = 0
    sum_image = None

    # List all image files in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Resize image if dimensions are provided
        if resize_dim:
            img = cv2.resize(img, resize_dim)

        # Convert to float for accurate summation
        img = img.astype(np.float32)

        # Initialize sum_image on the first valid image
        if sum_image is None:
            sum_image = np.zeros_like(img)

        # Accumulate pixel values
        sum_image += img
        total_images += 1

    if total_images == 0:
        raise ValueError("No valid images found in the folder.")

    # Compute average
    avg_image = sum_image / total_images

    # Convert back to uint8 for saving
    avg_image = np.clip(avg_image, 0, 255).astype(np.uint8)

    # Save the average image
    cv2.imwrite(output_path, avg_image)
    print(f"Average image saved to {output_path}")

# Example usage
resize_dim = (256, 256)  # Optional: Resize all images to 256x256
compute_average_image(image_folder, output_path, resize_dim)