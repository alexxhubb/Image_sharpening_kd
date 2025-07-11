# 📸 Image Sharpening using Knowledge Distillation

## 📖 Overview

This project implements an **image sharpening system using Knowledge Distillation (KD)**. A pretrained **SwinIR Super-Resolution model** acts as the **teacher network**, and a lightweight **CNN student model** learns to enhance blurred images to sharp images by mimicking the teacher’s outputs as well as ground truth.

The entire project was **built and trained on Kaggle**, and later modularized for portability on **VS Code** and GitHub.


## 📚 Technologies & Libraries Used

- **Python 3.10**
- **PyTorch** — deep learning framework
- **timm** — pretrained model utilities (for SwinIR)
- **Torchvision** — image transforms and datasets
- **scikit-image** — SSIM image quality metric
- **matplotlib** — image visualization
- **NumPy**
- **Pillow** — image handling
- **Kaggle environment** with GPU (T4)



## 🏆 Pretrained Teacher Model

- 📌 **Model Name:** SwinIR-M for Classical Image Super-Resolution  
- 📌 **Details:** Lightweight Swin Transformer based model for super-resolution, trained on DF2K dataset.  
- 📌 **Pretrained weights file:**  
  [`001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth)  

- 📌 **Official SwinIR Repo:** https://github.com/JingyunLiang/SwinIR  



## 🧠 Student Model

A custom lightweight CNN architecture with three convolutional layers and ReLU activations, optimized for image sharpening on low-resource hardware.

**Key components:**
- 3×3 Conv layers
- 64 filters
- ReLU activations
- Output: sharpened image matching the blurred input size



## 📂 Project File Structure

image-sharpening-kd/
├── README.md
├── requirements.txt
├
│
├── data/
│   ├── blur_dataset/
│   │   ├── blur_Image/
│   │   └── sharp_Image/
│
├── checkpoints/
│   ├── 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
│   └── student_model.pth
│
├── models/
│   ├── student_model.py
│   └── teacher_model.py
│
├── utils/
│   ├── dataset_loader.py
│   └── metrics.py
│
├── notebooks/
│   └── image_sharpening_kd.ipynb
│
├── outputs/
│   └── comparison_output.png
│
├── train_student.py
└── LICENSE




## 📊 SSIM (Structural Similarity Index Measure)

- 📌 **What is SSIM?**  
  It measures the perceptual similarity between two images (in terms of luminance, contrast, and structure) on a scale of 0 to 1 (or 0 to 100%).

- 📌 **How it’s used here:**  
  SSIM was computed between:
  - Student output vs. Ground truth sharp image
  - Higher SSIM implies better sharpness preservation.

- 📌 **Achieved SSIM:**  
  **95.30%** between student outputs and ground truth images.


## 🚀 How to Run This Project

### 📦 Install Dependencies


pip install -r requirements.txt 

📓 Run via Notebook (Kaggle / VS Code Jupyter)
Open notebooks/image_sharpening_kd.ipynb and run all cells sequentially.

🐍 Run via Python Script (VS Code)
python train_student.py

🖼️ Output Visualizations
Comparison of:

Blurred Input Image

Teacher SwinIR Output

Student CNN Output

Ground Truth Sharp Image

📌 Built On
✅ Kaggle GPU environment (T4) for model training

✅ VS Code for code modularization, cleanup and GitHub publishing

⭐ If you find this project helpful, consider starring it on GitHub!

