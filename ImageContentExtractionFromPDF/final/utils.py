from PIL import Image
from langchain_core.documents import Document
import os
import io
import base64

directory = "./figures/"

def image_to_base64(image_path):
    """Converts an image to base64."""
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')


def load_images_from_directory():
    """Loads images from a directory."""
    images = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img_path = os.path.join(directory, filename)
            print(os.path.join(directory, filename))
            images.append(img_path)
    return images


def load_img_to_image_str():
    """Loads images from a directory and converts them to base64 strings."""
    images = load_images_from_directory()
    image_str = []
   
    if not images:
        print("no images!")
        return False

    for img in images:
        image_str.append(image_to_base64(img))
    
    return image_str

def element_to_document(element):
    if isinstance(element, CompositeElement):
        content = element.text  # Assuming CompositeElement has a text attribute
    elif isinstance(element, Table):
        content = str(element.to_dict())  # Assuming Table has a to_list method to convert to a list
    else:
        content = str(element)  # Fallback to string representation for unknown types
    return Document(page_content=content)

