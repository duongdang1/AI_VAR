import torch 
import os
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from VAR_CNN import VarCNN
from CustomDataset import CustomDataset
from dataprep import create_dataloaders

model = VarCNN()
train_dir = 'model_data/train'
test_dir = 'model_data/test'

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

custom_dataset = CustomDataset(root_dir=train_dir, transform=data_transform)
print(f"Number of samples in CustomDataset: {len(custom_dataset)}")

batch_size = 32
num_workers = os.cpu_count()
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=batch_size,
    num_workers=num_workers
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
num_epochs =10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs,labels in train_dataloader: 
        outputs = model(inputs)
        loss = criterion(outputs,labels.unsqueeze(1).float())

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}")

# evaluate model 
model.eval()
with torch.no_grad():
    correct = 0 
    total = 0 
    for inputs,labels in test_dataloader:
        outputs = model(inputs)
        predictions = (outputs >= 0.5).float()
        total+= labels.size()
        correct += (predictions == labels.unsqueeze(1).float()).sum().item()
    accuracy = correct/total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
