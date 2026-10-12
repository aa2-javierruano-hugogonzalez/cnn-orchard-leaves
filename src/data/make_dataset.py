import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_SIZE = (96, 96)
TARGET_SAMPLES_PER_CLASS = 400
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def load_rgb_image(image_path):
    with Image.open(image_path) as image:
        return image.convert("RGB")


def split_dataset_by_class(data_dir, train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO, seed=RANDOM_SEED):
    random_generator = random.Random(seed)
    data_path = Path(data_dir)
    class_names = sorted([folder.name for folder in data_path.iterdir() if folder.is_dir()])

    train_samples = []
    validation_samples = []
    test_samples = []

    for label, class_name in enumerate(class_names):
        class_folder = data_path / class_name
        image_paths = sorted([path for path in class_folder.iterdir() if path.suffix.lower() == ".tif"])
        random_generator.shuffle(image_paths)

        num_images = len(image_paths)
        num_train = round(num_images * train_ratio)
        num_validation = round(num_images * validation_ratio)

        train_paths = image_paths[:num_train]
        validation_paths = image_paths[num_train:num_train + num_validation]
        test_paths = image_paths[num_train + num_validation:]

        train_samples += [(image_path, label) for image_path in train_paths]
        validation_samples += [(image_path, label) for image_path in validation_paths]
        test_samples += [(image_path, label) for image_path in test_paths]

    return train_samples, validation_samples, test_samples, class_names


class OrchardDataset(Dataset):
    def __init__(self, samples, transform=BASE_TRANSFORM):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = load_rgb_image(image_path)
        return self.transform(image), label


class BalancedOrchardDataset(Dataset):
    def __init__(self, samples, target_samples=TARGET_SAMPLES_PER_CLASS, seed=RANDOM_SEED,
                 base_transform=BASE_TRANSFORM, augment_transform=AUGMENT_TRANSFORM):
        self.base_transform = base_transform
        self.augment_transform = augment_transform

        self.image_paths = []
        self.labels = []
        self.is_augmented = []

        self._balance_classes(samples, target_samples, seed)

    def _balance_classes(self, samples, target_samples, seed):
        random_generator = random.Random(seed)

        paths_by_label = {}
        for image_path, label in samples:
            paths_by_label.setdefault(label, []).append(image_path)

        for label, real_paths in paths_by_label.items():
            for image_path in real_paths:
                self._add_sample(image_path, label, is_augmented=False)

            num_missing = target_samples - len(real_paths)
            for _ in range(num_missing):
                cloned_path = random_generator.choice(real_paths)
                self._add_sample(cloned_path, label, is_augmented=True)

    def _add_sample(self, image_path, label, is_augmented):
        self.image_paths.append(image_path)
        self.labels.append(label)
        self.is_augmented.append(is_augmented)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_rgb_image(self.image_paths[index])

        if self.is_augmented[index]:
            image = self.augment_transform(image)
        else:
            image = self.base_transform(image)

        return image, self.labels[index]


def get_dataloaders(data_dir, batch_size=32, num_workers=2, seed=RANDOM_SEED):
    train_samples, validation_samples, test_samples, class_names = split_dataset_by_class(data_dir, seed=seed)

    train_dataset = BalancedOrchardDataset(train_samples, seed=seed)
    validation_dataset = OrchardDataset(validation_samples)
    test_dataset = OrchardDataset(test_samples)

    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=shuffle_generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader, class_names


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "raw" / "leafscans-orchard-compressed"

    train_loader, validation_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size=16)

    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {tuple(images.shape)}")
    print(f"Labels batch shape: {tuple(labels.shape)}")
