import gdown
import os

# Define the URL of the file on Google Drive
file_url = 'https://drive.google.com/file/d/1-4ADjZJ9FpElHZKVk_-kQTBQpCVM80V5/download'

# Output path where the file will be saved
output_path = './category_classification_model-light.h5'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Download the file
gdown.download(file_url, output_path, quiet=False)