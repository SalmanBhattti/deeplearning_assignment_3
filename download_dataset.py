# run this once to download the dataset
from torchvision.datasets import Caltech101
dataset = Caltech101(root="./data", download=True)
print("Downloaded! Total images:", len(dataset))