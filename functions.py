import torch
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode, v2
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


def transform_set(set_type: str):
    """
    This function applies image transformations based on the dataset type and using predefined parameters.

        Args:
            set_type (str): The type of dataset for which transformations are applied.
                training: Applies random horizontal flip, rotation, color jitter, random channel permutation and resized crop for data augmentation (predefined parameters).
                validation: Applies normalization for preparing the data.

        Returns:
            torchvision.transforms.Compose: A composed transformation object.
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode, v2

    # Preprocessing for a training set
    if set_type == "training":
        transformation = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            
            # Data augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=.3, contrast = 0.3),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            v2.RandomChannelPermutation(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # Preprocessing without data augmentation (suitable for validation sets)
    elif set_type == "validation":
        transformation = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    return transformation

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) ##added 20-06
        self.dropout = nn.Dropout(0.1)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)  # Dropout with 25% probability
        self.dropout2 = nn.Dropout(0.5)   # Dropout with 50% probability

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        # self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Assuming input images are 32x32
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Assuming input images are 224x224
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # 1 output neuron
        self.fc3 = nn.Linear(128, 1)  # 1 output neuron for binary classification

    def forward(self, x): # Input x is passed through each layer sequentially
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x))) ##added 20-06
        x = self.pool(F.relu(self.conv1(x)))  # First conv layer + ReLU + max pooling
        x = self.dropout1(x)  # Dropout after first conv layer

        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer + ReLU + max pooling
        x = self.dropout1(x)  # Dropout after second conv layer

        x = self.pool(F.relu(self.conv3(x)))  # Third conv layer + ReLU + max pooling
        x = self.dropout1(x)  # Dropout after third conv layer

        # Flatten the tensor before the fully connected layers
        x = x.view(-1, 128 * 28 * 28)  # Flattening / Reshapes the output of the second convolutional layer to be compatible with the fully connected layers
        x = x.view(-1, 128 * 28 * 28)  # Flatten the output of the third conv layer

        # Fully connected layers with ReLU
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Dropout before the first fully connected layer
        x = F.relu(self.fc1(x))  # First fully connected layer + ReLU
        x = self.dropout2(x)  # Dropout before the second fully connected layer
        x = F.relu(self.fc2(x))  # Second fully connected layer + ReLU
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification

        # Fully connected layers with ReLU
	@@ -95,9 +118,11 @@ def forward(self, x): # Input x is passed through each layer sequentially
        x = self.dropout(x)
        x = F.relu(self.fc3(x))        
        x = torch.sigmoid(self.fc4(x))  # Sigmoid activation for binary classification

        return x

model = ConvNet()
print(model)  

# Function to evaluate the model
def evaluate_model(model, data, labels, metric):
    """
        Evaluates the trained neural network based on the chosen evaluation metric.
        
        Args:
        model: Trained model
        data: Input data the model was trained on.
        labels: The labels the model should predict.
        metric: Evaluation metric
            accuracy: Measures how often the model correctly predicts the outcome. 
            f1: Measures the model's accuracy combining the precision and recall scores of the model. 
    """
    with torch.no_grad():
        output = model(data)
        predictions = (output > 0.5).float()  # Apply threshold to convert probabilities to binary predictions
        if metric == "accuracy":
            return accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        elif metric == "f1":
            return f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        else:
            raise ValueError(f"Invalid metric: {metric}")


def train_model(model, train_loader, val_loader, n_epochs=20, learning_rate=0.001):
    train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s = [], [], [], [], [], []

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):  # Loop over the dataset multiple times
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_f1 = 0.0
        total_train_batches = len(train_loader)
        
        for train_images, train_labels in train_loader:
            # Forward pass
            outputs = model(train_images)
            
            # Reshape labels to match the output shape
            train_labels = train_labels.view(-1, 1).float()

            # Compute loss
            loss = criterion(outputs, train_labels)
            
            # Backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # Calculate accuracy and F1-score
            predictions = (outputs > 0.5).float()  # Apply threshold to convert probabilities to binary predictions
            train_acc += accuracy_score(train_labels.cpu().numpy(), predictions.cpu().numpy())
            train_f1 += f1_score(train_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        
        # Calculate average training loss, accuracy, and F1-score for the epoch
        avg_train_loss = train_loss / total_train_batches
        avg_train_acc = train_acc / total_train_batches
        avg_train_f1 = train_f1 / total_train_batches
        
        # Append average training metrics for this epoch
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        train_f1s.append(avg_train_f1)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        total_val_batches = len(val_loader)
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                outputs = model(val_images)
                val_labels = val_labels.view(-1, 1).float()
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()
                
                # Calculate accuracy and F1-score
                predictions = (outputs > 0.5).float()  # Apply threshold to convert probabilities to binary predictions
                val_acc += accuracy_score(val_labels.cpu().numpy(), predictions.cpu().numpy())
                val_f1 += f1_score(val_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        
        # Calculate average validation loss, accuracy, and F1-score for the epoch
        avg_val_loss = val_loss / total_val_batches
        avg_val_acc = val_acc / total_val_batches
        avg_val_f1 = val_f1 / total_val_batches
        
        # Append average validation metrics for this epoch
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        val_f1s.append(avg_val_f1)
        
        print(f'Epoch [{epoch+1}/{n_epochs}] - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Training Accuracy: {avg_train_acc:.4f}, Validation Accuracy: {avg_val_acc:.4f}, Training F1: {avg_train_f1:.4f}, Validation F1: {avg_val_f1:.4f}')

    print('Training complete')


    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(range(n_epochs), train_losses, '--', label='Training Loss')
    plt.plot(range(n_epochs), val_losses, label='Validation Loss')
    plt.xticks(range(n_epochs), range(n_epochs))
    plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.plot(range(n_epochs), avg_train_acc, '--', label='Training Accuracy')
    # plt.plot(range(n_epochs), avg_val_acc, label='Validation Accuracy')
    # plt.xticks(range(n_epochs), range(n_epochs))
    # plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(range(n_epochs), avg_train_f1, '--', label='Training F1 Score')
    # plt.plot(range(n_epochs), avg_val_f1, label='Validation F1 Score')
    # plt.xticks(range(n_epochs), range(n_epochs))
    # plt.legend()

    plt.show()
