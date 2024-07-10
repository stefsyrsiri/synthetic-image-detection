_import torch
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode, v2
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import random
import os
import sys
from PIL import Image
import time
import pandas as pd
import numpy as np
import tempfile
from typing import Dict, List, Tuple
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import torchvision.transforms.v2 as v2_transforms
import cv2
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def transform_set(set_type: str):
    """
    This function applies image transformations based on the dataset type and using predefined parameters.

        Args:
            set_type (str): The type of dataset for which transformations are applied.
                training: Applies random horizontal flip, rotation, color jitter, random channel permutation and resized crop for data augmentation (predefined parameters).
                validation: Applies basic transformation without augmentation.
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
             transforms.RandomRotation(30),
             transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
             v2.RandomChannelPermutation(),
             transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    #Preprocessing without data augmentation (suitable for validation sets)
    elif set_type == "validation":
        transformation = transforms.Compose([
             transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ])

    return transformation
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Neural net architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)  # Dropout with 25% probability
        self.dropout2 = nn.Dropout(0.5)   # Dropout with 50% probability

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Input images are 224x224 (training set transformation)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # 1 output neuron for binary classification

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # First conv layer + ReLU + max pooling
        x = self.dropout1(x)  # Dropout after first conv layer

        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer + ReLU + max pooling
        x = self.dropout1(x)  # Dropout after second conv layer

        x = self.pool(F.relu(self.conv3(x)))  # Third conv layer + ReLU + max pooling
        x = self.dropout1(x)  # Dropout after third conv layer

        # Flatten the tensor before the fully connected layers
        x = x.view(-1, 128 * 28 * 28)  # Flatten the output of the third conv layer

        # Fully connected layers with ReLU
        x = self.dropout2(x)  # Dropout before the first fully connected layer
        x = F.relu(self.fc1(x))  # First fully connected layer + ReLU
        x = self.dropout2(x)  # Dropout before the second fully connected layer
        x = F.relu(self.fc2(x))  # Second fully connected layer + ReLU
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification

        return x
        
#-------------------------------------------------------------------------------------------------------------------------------------------------------------  

def train_model(model, train_loader, val_loader, criterion, weight_decay=1e-5, n_epochs=20, learning_rate=0.001, patience=5, tl=False):
    """
    This function trains a model given a neural network architecture. It uses slightly different argument values based on whether it is a model being trained from the start or a pre-trained model being fine-tuned. At the end of the training a loss and f1-score charts are plotted.

        Args:
            model: A model with a predefined architecture to be trained.
            train_loader: The training set data loader.
            val_loader: The validation set data loader.
            criterion: The criterion based on which the model will evaluate its answers (e.g. Cross Entropy Loss)
            weight_decay (float): A penalty for the model's weights for reguralization. Default is 1e-5.
            n_epochs (int): The number of times (epochs) the model will loop through the entire dataset for training. Default is 20.
            learning_rate (float): The rate the model is being trained at. Default is 0.001.
            patience (int): The number of epochs the model should wait for the improvement of the average validation loss. Default is 5.
            tl (bool): A boolean that sets whether the training session regards transfer learning. Default is False.
    """
    train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s = [], [], [], [], [], []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if tl:
      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Check if GPU is available and move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_params = model.state_dict()

    since = time.time()

    # Training loop
    for epoch in range(n_epochs):  # Corrected range and print statement
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_f1 = 0.0
        total_train_batches = len(train_loader)

        for train_images, train_labels in train_loader:
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            optimizer.zero_grad()
            outputs = model(train_images)
            train_labels = train_labels.view(-1, 1).float()
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_acc += accuracy_score(train_labels.cpu().numpy(), predictions.cpu().numpy())
            train_f1 += f1_score(train_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')

        avg_train_loss = train_loss / total_train_batches
        avg_train_acc = train_acc / total_train_batches
        avg_train_f1 = train_f1 / total_train_batches

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
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                val_labels = val_labels.view(-1, 1).float()
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()

                predictions = (outputs > 0.5).float()
                val_acc += accuracy_score(val_labels.cpu().numpy(), predictions.cpu().numpy())
                val_f1 += f1_score(val_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')

        avg_val_loss = val_loss / total_val_batches
        avg_val_acc = val_acc / total_val_batches
        avg_val_f1 = val_f1 / total_val_batches

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        val_f1s.append(avg_val_f1)

        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch [{epoch+1}/{n_epochs}] - '
            f'Learning Rate: {current_lr:.6f}, '
            f'Training Loss: {avg_train_loss:.4f}, '
            f'Validation Loss: {avg_val_loss:.4f}, '
            f'Training Accuracy: {avg_train_acc:.4f}, '
            f'Validation Accuracy: {avg_val_acc:.4f}, '
            f'Training F1: {avg_train_f1:.4f}, '
            f'Validation F1: {avg_val_f1:.4f}'
        )

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Save the model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_params = model.state_dict()

            # Save the model state dictionary to a file
            save_path = './BestParameters'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(best_model_params, os.path.join(save_path, 'best_model.pth'))
            print(f'Saving model with validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs.')
                break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation loss: {best_val_loss:.4f}')

    # Load the best model parameters
    model.load_state_dict(best_model_params)
    epochs_range = range(1, len(train_losses) + 1)

    # Plotting training progress
    plt.figure(figsize=(10, 8))

    # Plotting Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_losses, '--', label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xticks(epochs_range)
    plt.legend()

    # Plotting F1-score
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_f1s, '--', label='Training F1-score')
    plt.plot(epochs_range, val_f1s, label='Validation F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.title('Training and Validation F1-score')
    plt.xticks(epochs_range)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

#-------------------------------------------------------------------------------------------------------------------------------------------------------------  

def evaluate_model(model, test_loader):
    """
    This function evaluates a trained model against a test data loader. At the end of the evaluation a confusion matrix and classification report are shown.

        Args:
            model: A model with a predefined architecture to be trained.
            test_loader: The test set data loader.
    """
    
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_f1 = 0.0
    predictions_all = []
    labels_all = []

    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = model(test_images)
            
            # Apply sigmoid activation to the outputs
            outputs = torch.sigmoid(outputs)
            test_labels = test_labels.view(-1, 1).float()
            loss = criterion(outputs, test_labels)
            test_loss += loss.item()

            predictions = (outputs > 0.5).float()
            predictions_all.extend(predictions.cpu().numpy())
            labels_all.extend(test_labels.cpu().numpy())

            test_acc += accuracy_score(test_labels.cpu().numpy(), predictions.cpu().numpy())
            test_f1 += f1_score(test_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    avg_test_f1 = test_f1 / len(test_loader)

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}, Test F1-score: {avg_test_f1:.4f}')

    # Convert lists to numpy arrays for confusion matrix calculation
    predictions_all = np.array(predictions_all)
    labels_all = np.array(labels_all)

    # Compute confusion matrix
    cm = confusion_matrix(labels_all, predictions_all)
    print('Confusion Matrix:')
    print(cm)

    # Plot confusion matrix with seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Print classification report
    target_names = ['Real', 'Fake']
    print('\nClassification Report:')
    print(classification_report(labels_all, predictions_all, target_names=target_names))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Predict labels for demo
def demo_predict(input_image):
    """
    This function transforms a given image following the exact transformation the validation set had during the training of the model and then predicts its label.

        Args:
            input_image: An image that is getting in the demo from the test set, from the user's webcam or device.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_image = transform(input_image).unsqueeze(0)
    input_image = input_image.to(device)

    with torch.no_grad():
        prediction = model_demo(input_image)
        confidence = torch.sigmoid(prediction).item()
        label = 'Fake' if confidence < 0.5 else 'Real'

    return label
