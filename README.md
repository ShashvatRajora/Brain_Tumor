# 🧠 Brain Tumor MRI Image Classification

This project focuses on classifying brain tumor types from MRI images using various deep learning models including custom CNNs, pretrained architectures (VGG16), and optimization techniques like **Grey Wolf Optimizer (GWO)** and **Sin Cosine Optimizer (SCO)**. It also includes a user-friendly **Streamlit app** to allow real-time tumor prediction from uploaded MRI images.

---

## 📂 Dataset

The dataset consists of MRI images of four brain tumor classes:

- `glioma`
- `meningioma`
- `pituitary`
- `no_tumor`

Each class contains grayscale or RGB MRI images of varying dimensions. The images are resized to **224×224** for training and evaluation.

---

## 🧠 Models Trained

### ✅ 1. Custom CNN Model
- 3 Convolutional layers
- BatchNorm + ReLU + MaxPooling
- Fully connected layers (512 → 256 → 4)
- Trained using Adam optimizer

### ✅ 2. VGG16 Transfer Learning
- Pretrained on ImageNet
- Custom classification head added
- Frozen base layers during initial training
- Fine-tuned with optimizer injections

### ✅ 3. Optimized Models
- VGG16 + **Grey Wolf Optimizer (GWO)**
- VGG16 + **Sin Cosine Optimizer (SCO)**
- Injected optimized weights into the dense layers of the classifier head to improve generalization.

---

## 📊 Evaluation

| Model              | Accuracy (%) | Notes                               |
|-------------------|--------------|-------------------------------------|
| Custom CNN         | ~60%         | Baseline performance                |
| VGG16 (Vanilla)    | ~69%         | Pretrained, no optimization         |
| VGG16 + GWO        | ~72%         | Optimized weights                   |
| VGG16 + SCO        | ~74%         | Best performance                    |

Metrics used:
- Accuracy
- Validation Loss
- Confusion Matrix
- Class-wise prediction confidence

---

## 📉 UBM Analysis (Univariate, Bivariate, Multivariate)

Performed statistical analysis on image properties:
- 📦 File size
- 🌕 Pixel mean & standard deviation
- 📈 Pairwise distributions by class

### Key Insights:
- Unequal brightness and size distributions across classes
- Class imbalance detected (more glioma and pituitary cases)
- Some tumors are more visually separable based on pixel stats

---

## 💡 Streamlit App (Real-time Prediction UI)

### 🎯 Features:
- Upload an MRI image
- Predict using all models
- View predicted class + confidence
- Visual bar chart of class scores

### ▶️ Run Locally:

```bash
streamlit run app.py
