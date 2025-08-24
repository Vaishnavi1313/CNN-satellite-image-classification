# 🌍 Satellite Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify satellite images into multiple land-cover categories (e.g., urban, vegetation, water, desert,fire).  
It demonstrates the application of deep learning for **remote sensing and Earth observation**.

---

## 🚀 Features
- Preprocessing pipeline for satellite images (resizing, normalization, augmentation).
- CNN architecture designed for image classification.
- Training and validation with accuracy/loss visualization.
- Evaluation metrics: accuracy, precision, recall, F1-score.
- Easy-to-use scripts for training and inference.

---

## 🗂 Dataset
- The model is trained on [**EuroSAT**](https://github.com/phelber/EuroSAT) / [UC Merced Land Use](http://weegee.vision.ucmerced.edu/datasets/landuse.html) dataset  
  (replace with the one you actually used).
- Images are RGB, resolution 64x64 / 256x256 pixels (depending on dataset).
- Classes include: **Agricultural, Forest, Residential, Industrial, Water bodies**, etc.

---

## 🔧 Tech Stack
- **Python 3.9+**
- **TensorFlow / PyTorch**
- NumPy, Pandas, Matplotlib
- Scikit-learn

---

## 📊 Results
- Achieved **85% accuracy** on the test set.
- Confusion matrix & classification report are included.
- Example predictions:

| Input Image | Predicted Label | True Label |
|-------------|-----------------|------------|
| [sample1](![4](https://github.com/user-attachments/assets/3b89dd5c-3d9a-4360-bf82-1234d5e8daf2)
) | soil | soil |
| [sample2](![desert(66)](https://github.com/user-attachments/assets/3686d6f1-44fc-4e07-925c-fedf61fbb56a)
) | desert | desert |
| [sample3](![Forest_22](https://github.com/user-attachments/assets/c873b54b-eb81-4dfd-a19a-4827ee6498bd)
) | forest | forest |

---

## 🛠 Installation & Usage
```bash
# Clone the repository
git clone https://github.com/Vaishnavi1313/CNN-satellite-image-classification.git
cd satellite-cnn

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py --epochs 20 --batch_size 32

# Evaluate
python src/evaluate.py --model saved_model.pth

