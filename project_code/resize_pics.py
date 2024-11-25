import os
from PIL import Image

# Define paths
input_folder = "projets_data/project_code/assets/img_celeba/"  # Folder containing original images
output_folder = "projets_data/celeba/img_celeba_resized/"  # Folder to save resized images

# Desired resolution
new_width = 150  # Adjust the width as needed
new_height = 150  # Adjust the height as needed

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # Skip non-image files
    if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
        print(f"Skipping non-image file: {filename}")
        continue

    try:
        # Open the image
        with Image.open(input_path) as img:
            # Resize the image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save the resized image
            img_resized.save(output_path)
            print(f"Resized and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("All images have been resized and saved.")
