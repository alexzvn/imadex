from PIL import Image
from Net.resnet18 import FeatureVector

vector = FeatureVector()

def make(image_path):
    return Image.open(image_path).convert('RGB')

def extract(image):
    return vector.get_vector(image).tolist()
