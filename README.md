# Fashion MNIST Classification with Neural Networks

This project implements a neural network-based classifier for the Fashion MNIST dataset using **TensorFlow Keras**. The goal is is to classify images of fashion items into 10 categories.

---

## **Project Overview**

We built a fully-connected neural network with three hidden layers to classify Fashion MNIST images. The model was trained and evaluated on the training, validation, and test datasets.

### Key Features

- Loading and preprocessing the Fashion MNIST dataset
- Building a customizable neural network
- Training with early stopping and learning rate reduction
- Evaluating the model on training, validation, and test sets
- Plotting training history (loss and accuracy curves)
- Plotting the confusion matrix
- Visualizing sample predictions

---

## **Project Structure**

\`\`\`
fashion-mnist-classification/
│
├── fashion_mnist_nn.py      # Main Python script with model, training, and evaluation
├── README.md                 # Project README file
└── venv/                     # Virtual environment (optional)
\`\`\`

---

## **Installation**

### 1. Clone the repository

\`\`\`bash
git clone <your-github-repo-link>
cd fashion-mnist-classification
\`\`\`

### 2. Create a virtual environment (optional but recommended)

\`\`\`bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
\`\`\`

### 3. Install required packages

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## **Usage**

Run the main script to train and evaluate the model:

\`\`\`bash
python fashion_mnist_nn.py
\`\`\`

### This will:

1. Load and preprocess the Fashion MNIST dataset
2. Build the neural network
3. Train the model with early stopping and learning rate reduction callbacks
4. Evaluate the model on training, validation, and test sets
5. Plot training history and confusion matrix
6. Visualize sample predictions

---

## **Hyperparameters**

We used the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Hidden layers | [256, 128, 64] |
| Activation function | ReLU |
| Dropout rate | 0.3 |
| Output activation | Softmax |
| Optimizer | Adam (learning rate=0.001, AMSGrad=True, clipnorm=1.0) |
| Loss function | Sparse Categorical Crossentropy |
| Batch size | 128 |
| Epochs | 20 |
| Callbacks | EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.2, patience=3) |

---

## **Results**

| Dataset | Accuracy |
|---------|----------|
| Training | 78.43% |
| Validation | 78.85% |
| Test | 77.15% |

### Key Observations

- The confusion matrix shows most misclassifications occur between visually similar items (e.g., Shirt vs T-shirt/top)
- The model generalizes well with similar validation and test accuracies
- Training and validation curves indicate minimal overfitting


## **Future Improvements**

- **Use CNNs**: Implementing a Convolutional Neural Network could improve accuracy by capturing spatial features in images
- **Hyperparameter tuning**: Experiment with different learning rates, layer sizes, and dropout rates
- **Data augmentation**: Apply image transformations to increase dataset diversity
- **Ensemble methods**: Combine multiple models for improved predictions
- **Advanced architectures**: Try ResNet, DenseNet, or other modern architectures

---

## **Requirements**

\`\`\`
tensorflow>=2.x
numpy
matplotlib
scikit-learn
seaborn
\`\`\`

---

## **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

---

## **Author**

Mona Ahmed Mohamed, Amna Khaled Ben Mousa,Hamdi Abdi Mohamed

---

## **Acknowledgments**

- Fashion MNIST dataset by Zalando Research
- TensorFlow and Keras teams for the excellent framework
