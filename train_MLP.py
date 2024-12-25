import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mnist_viewer import load_mnist_images, load_mnist_labels
from math import ceil
import pickle


### PATHS
origin_path = ""
model_num = 2
model_file_path = origin_path + f"models/mlp{model_num}.pth"
images_path = origin_path + "dataset/train-images-idx3-ubyte/train-images-idx3-ubyte"
labels_path = origin_path + "dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
train_stats_path = origin_path + "train_stats.pkl"


### CONSTANTS
dataset_size = 60000
img_size = 28
input_dim = img_size * img_size
output_dim = 10


### HYPER-PARAMETERS
learning_rate = 0.00005
batch_size = 32
epochs = 300
num_of_batches = ceil(dataset_size / batch_size)


### MLP ARCHITECTURE
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.model(x)


### TENSOR CREATION and CONVERTION 
# tensor --> label
def tensor2label(output_tensor):
    max_ind = 0
    for i in range(1, output_dim):
        if output_tensor[max_ind] < output_tensor[i]:
            max_ind = i
    return max_ind

# label --> tensor
def label2tensor(label):
    array = np.zeros(output_dim)
    array[label] = 1.0
    return torch.tensor(array, dtype=torch.float32)

# Dataset image input tensor creation
def input_image_tensor(batch_idx, batch_size):
    offset = batch_idx*batch_size
    image_np_array = images[offset:offset+batch_size]
    tensor = torch.tensor(image_np_array, dtype=torch.float32)
    tensor /= 255
    return tensor

# Dataset Label input tensor creation
def input_label_tensor(batch_idx, batch_size):
    offset = batch_idx*batch_size
    label_np_array = labels[offset:offset+batch_size]
    # label_np_array = np.vectorize(label2tensor)(label_np_array)  # One-hot encoding (not required)
    tensor = torch.tensor(label_np_array)
    return tensor


### MLP Training Context
# Models
mlp = MLP(input_dim=input_dim, output_dim=output_dim)

# Loss Function
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer_mlp = optim.Adam(mlp.parameters(), lr=learning_rate)


### TRAINING THE MODEL
# Loading entire dataset
images = load_mnist_images(images_path)
labels = load_mnist_labels(labels_path)

# Training Loop
mlp_loss_history = np.zeros(epochs)

for epoch in range(epochs):
    # Training the mlp layers
    for batch_idx in range(num_of_batches):
        image_tensor = input_image_tensor(batch_idx, batch_size)
        label_tensor = input_label_tensor(batch_idx, batch_size)
        
        optimizer_mlp.zero_grad()
        output = mlp(image_tensor)
        mlp_loss = loss_function(output, label_tensor)
        mlp_loss.backward()
        optimizer_mlp.step()

    # Storing loss values after every epoch
    mlp_loss_val = mlp_loss.item()
    print(f"Completed: {epoch}/{epochs}  Loss: {mlp_loss_val: .4f}")
    mlp_loss_history[epoch] = mlp_loss_val


### SAVING MODEL WEIGHTS and STATISTICS
# Save built and trained mlp model
torch.save(mlp.state_dict(), model_file_path)

# Save training statistics
with open("train_stats.pkl", "wb") as file:
    pickle.dump(mlp_loss_history, file)
