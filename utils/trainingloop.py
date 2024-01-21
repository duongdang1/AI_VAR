# engine.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.EarlyStopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_device():
    # Check if GPU is available, otherwise use CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_dataloader, test_dataloader, num_epochs=50, lr=0.001):
    device = get_device()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=10, delta=0, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience = 5, factor =0.1, verbose = True)
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc":[],
    }
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate training accuracy
            predictions = (outputs >= 0.5).float()
            correct_train += (predictions == labels.unsqueeze(1).float()).sum().item()
            total_train += labels.size(0)

        average_loss = total_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train

        # Testing
        model.eval()
        with torch.no_grad():
            total_loss_test = 0.0
            correct_test = 0
            total_test = 0

            for inputs_test, labels_test in test_dataloader:
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)

                outputs_test = model(inputs_test)
                loss_test = criterion(outputs_test, labels_test.unsqueeze(1).float())
                total_loss_test += loss_test.item()

                #Calculate testing accuracy
                predictions_test = (outputs_test >= 0.5).float()
                correct_test += (predictions_test == labels_test.unsqueeze(1).float()).sum().item()
                total_test += labels_test.size(0)

            average_loss_test = total_loss_test / len(test_dataloader)
            test_accuracy = correct_test / total_test

        results["train_loss"].append(average_loss)
        results['train_acc'].append(train_accuracy)
        results["test_loss"].append(average_loss_test)
        results['test_acc'].append(test_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {average_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Test Loss: {average_loss_test:.4f}, "
              f"Test Acc: {test_accuracy:.4f}")
        if early_stopping(average_loss_test): 
            print("early stopping")
            break
        scheduler.step(average_loss_test)
        
    return results