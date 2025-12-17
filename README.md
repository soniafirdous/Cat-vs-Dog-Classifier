# 🐱🐶 Cat vs Dog Image Classification using CNN & TensorFlow

## 📌 Project Overview
This project builds a **binary image classification system (Cat vs Dog)** using **TensorFlow and CNNs**. The complete deep learning pipeline is implemented using **tf.data.Dataset** for efficient data handling and **regularization techniques** to reduce overfitting.

---

## 📂 Dataset
- **Dataset:** Cats vs Dogs
- **Source:** Kaggle
- **Classes:** Cat, Dog
- **Image Size:** 224 × 224 × 3

---

## 🔄 Project Workflow

### 1️⃣ Download Dataset
- Dataset downloaded from **Kaggle**
- Extracted into training and validation directories

---

### 2️⃣ Import Libraries
Key libraries used:
- TensorFlow / Keras
- NumPy
- Matplotlib

---

### 3️⃣ Load Images using `tf.data.Dataset`
- Images loaded efficiently using TensorFlow’s data pipeline
- Enables fast and scalable training

---

### 4️⃣ Create Data Augmentation Layers
Applied to training data to improve generalization:
- Random Flip
- Random Rotation
- Random Zoom

---

### 5️⃣ Create Preprocessing Functions
- Resize images to 224 × 224
- Normalize pixel values
- Apply augmentation only on training data

---

### 6️⃣ Data Pipeline Optimization
Applied in sequence:
```text
map → shuffle → prefetch
Improves training speed

Prevents I/O bottlenecks

7️⃣ Build CNN Model (Anti-Overfitting)

Model includes:

Convolutional layers

Batch Normalization

Dropout layers

Global Average Pooling (instead of Flatten)

8️⃣ Compile the Model

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

9️⃣ Add Early Stopping

Monitors validation loss

Stops training when performance stops improving

Prevents overfitting

🔟 Train the Model

Model trained on augmented data

Validation performed on unseen images

🏗️ Model Architecture
Input Image
↓
Conv2D + BatchNorm + ReLU
↓
MaxPooling
↓
Conv2D + BatchNorm + ReLU
↓
Global Average Pooling
↓
Dropout
↓
Dense (Sigmoid)

📊 Evaluation Metrics

Accuracy

Training vs Validation Loss

Overfitting Analysis

🚀 Key Learnings

tf.data.Dataset improves performance and scalability

Data augmentation reduces overfitting

EarlyStopping prevents unnecessary training

Global Average Pooling reduces parameters

🛠️ Tech Stack

Python | TensorFlow | Keras | CNN | tf.data | Deep Learning

👩‍💻 Author

Sonia Firdous
Data Science & Deep Learning
