import os
from pdf2image import convert_from_path

# Path to your PDF
pdf_path = "my_doccc.pdf"  # ← Change this to your actual PDF path

# Output folder
output_folder = "text_sig_dataset"

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert PDF to images
images = convert_from_path(pdf_path)

# Save each image with numbered filename
for i, image in enumerate(images, start=1):
    image_path = os.path.join(output_folder, f"image_{i}.png")
    image.save(image_path, "PNG")

print(f"Saved {len(images)} images to '{output_folder}' folder.")
