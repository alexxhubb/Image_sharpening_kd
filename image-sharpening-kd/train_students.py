# 📦 Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

from models.student_model import StudentCNN
from models.teacher_model import load_teacher_model
from utils.dataset_loader import BlurSharpDataset

# 📌 Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Data Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 📌 Dataset & DataLoader
train_dataset = BlurSharpDataset(
    blur_dir="data/blur_dataset/blur_Image",
    sharp_dir="data/blur_dataset/sharp_Image",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 📌 Load Models
student_model = StudentCNN().to(device)
teacher_model = load_teacher_model("checkpoints/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth").to(device)

# 📌 Loss & Optimizer
criterion_l1 = nn.L1Loss()
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# 📌 Training Loop
student_model.train()
for epoch in range(10):
    running_loss = 0.0
    for blur_img, sharp_img in train_loader:
        blur_img, sharp_img = blur_img.to(device), sharp_img.to(device)

        student_output = student_model(blur_img)
        teacher_output = teacher_model(blur_img)
        teacher_output_resized = F.interpolate(teacher_output, size=(256, 256), mode='bilinear', align_corners=False)

        loss_l1 = criterion_l1(student_output, sharp_img)
        kd_loss = criterion_l1(student_output, teacher_output_resized)
        total_loss = loss_l1 + 0.5 * kd_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# 📌 Save Trained Student Model
torch.save(student_model.state_dict(), "checkpoints/student_model.pth")
print("Student model saved successfully.")

# 📌 SSIM Function (as you had it)
def calculate_ssim(img1, img2):
    img1 = img1.permute(1, 2, 0).detach().cpu().numpy()
    img2 = img2.permute(1, 2, 0).detach().cpu().numpy()
    return ssim(img1, img2, data_range=1.0, channel_axis=-1)

# 📌 Evaluate SSIM on One Example
student_model.eval()
with torch.no_grad():
    for blur_img, sharp_img in train_loader:
        blur_img, sharp_img = blur_img.to(device), sharp_img.to(device)
        student_output = student_model(blur_img)
        ssim_value = calculate_ssim(student_output[0], sharp_img[0])
        print(f"Final SSIM: {ssim_value*100:.2f}%")
        break
