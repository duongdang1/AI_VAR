import os 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image 

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform = None): 
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = self.load_data()

    def load_data(self):
        data = []
        for cls in self.classes: 
            class_path = os.path.join(self.root_dir, cls)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path,filename)
                data.append((img_path,self.class_to_idx[cls]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label