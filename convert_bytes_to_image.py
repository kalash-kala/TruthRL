import io
from PIL import Image
import ast

def convert(byte_str):
    image = Image.open(io.BytesIO(byte_str))
    image.save("/home/kalashkala/TruthRL/output_image.jpg")
    print("Image saved to /home/kalashkala/TruthRL/output_image.jpg")

if __name__ == "__main__":
    # Assuming you paste the exact string representation of the dictionary
    data = {'bytes': b'\xff\xd8\xff\xe0\x00\x10JFIF...'} # truncated for script
    
