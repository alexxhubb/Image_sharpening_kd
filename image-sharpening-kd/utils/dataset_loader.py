import torch
from torch.utils.data import Dataset
from PIL import Image
import os

# Custom PyTorch dataset for blur and sharp image pairs
class BlurSharpDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, transform=None):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.image_names = os.listdir(blur_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.image_names[idx])
        sharp_path = os.path.join(self.sharp_dir, self.image_names[idx])

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img
