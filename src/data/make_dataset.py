import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BalancedOrchardDataset(Dataset):
    """
    Custom PyTorch Dataset that balances classes dynamically.
    Minority classes are oversampled up to 'target_samples', 
    and the duplicated samples receive heavy data augmentation.
    """
    def __init__(self, root_dir, target_samples=400, transform=None, augment_transform=None):
        self.root_dir = Path(root_dir)
        self.target_samples = target_samples
        self.transform = transform
        self.augment_transform = augment_transform
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        self.is_augmented = []
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            img_files = [p for p in cls_dir.iterdir() if p.suffix.lower() == '.tif']
            
            num_real = len(img_files)
            if num_real == 0:
                continue
                
            label_idx = self.class_to_idx[cls_name]
            
            for img_path in img_files:
                self.image_paths.append(img_path)
                self.labels.append(label_idx)
                self.is_augmented.append(False)
            
            if num_real < self.target_samples:
                num_synthetic = self.target_samples - num_real
                for _ in range(num_synthetic):
                    random_img = random.choice(img_files)
                    self.image_paths.append(random_img)
                    self.labels.append(label_idx)
                    self.is_augmented.append(True) 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        is_aug = self.is_augmented[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if is_aug and self.augment_transform:
            image = self.augment_transform(image)
        elif self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(data_dir, batch_size=32, target_samples=400):
    """
    Builds the PyTorch DataLoader with the appropriate transforms.
    """
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BalancedOrchardDataset(
        root_dir=data_dir,
        target_samples=target_samples,
        transform=base_transform,
        augment_transform=aug_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader, dataset.classes

if __name__ == '__main__':
    DATA_DIR = 'data/raw/leafscans-orchard-compressed' if os.path.exists('data/raw/leafscans-orchard-compressed') else 'data/raw'
    
    loader, class_names = get_dataloaders(DATA_DIR, batch_size=16)
    print(f"Classes detected: {class_names}")
    print(f"Total images in balanced dataset: {len(loader.dataset)} (Should be exactly {len(class_names) * 400})")
    
    images, labels = next(iter(loader))
    print(f"\nBatch images shape: {images.shape} # (batch_size, channels, height, width)")
    print(f"Batch labels shape: {labels.shape}")