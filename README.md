
# MNIST Handwritten Digits Classification with MLP

This repository contains a Python implementation of a Multi-Layer Perceptron (MLP) for classifying handwritten digits from the MNIST dataset. The model is built, trained, tested, and used for predictions in a straightforward workflow, making it ideal for beginners and enthusiasts.

## Features

- **Dataset**: Uses the MNIST handwritten digits dataset (60,000 training images and 10,000 testing images).
- **Model Architecture**: A fully connected neural network (MLP) designed for digit classification.
- **Training**: Model is trained using standard supervised learning techniques.
- **Evaluation**: Achieves a reasonably high accuracy of ~(97.5-98)% on the test dataset.
- **Prediction**: Can predict handwritten digits from custom input images.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python (>=3.8)
- `numpy`
- `pytorch`
- `pillow`
- `matplotlib`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/i-am-raahul-m/MLP-Mnist-Handwritten-Digits.git
   cd MLP-Mnist-Handwritten-Digits
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Training the Model

Run the training script to train the MLP:

```bash
python train_MLP.py
```

The script will:
- Load the MNIST dataset.
- Normalize and preprocess the images.
- Build and train the MLP model.
- Save the trained model to a file (`models/mlp_.pth`).

### 2. Testing the Model

Evaluate the model's accuracy on the test dataset:

```bash
python test_MLP.py
```

This script loads the saved model and computes accuracy performance metric on the test dataset.

### 3. Making Predictions

Use the trained model to predict custom images:

```bash
python predict_MLP.py 
```

---

Specify the path of the image file to be predicted in the "image_path" global variable

## Model Architecture

- **Input Layer**: 28x28 pixels flattened to a vector of size 784.
- **Hidden Layers 1**: 128 neurons, fully connected layer with Leaky ReLU activation.
- **Hidden Layers 2**: 128 neurons, fully connected layer with Leaky ReLU activation.
- **Hidden Layers 3**: 128 neuron, fully connected layer with Leaky ReLU activation.
- **Output Layer**: 10 neurons (one for each digit).

---

### Sample Predictions

Here is a sample prediction made by the model:

|     Input Image     | Predicted Label |
|---------------------|-----------------|
| ![handwritten_digit](custom_handwritten_digit.png) |     Digit 8      |

---

## Technologies Used

- PyTorch
- NumPy
- Matplotlib
- Python Imaging Library (PIL)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
