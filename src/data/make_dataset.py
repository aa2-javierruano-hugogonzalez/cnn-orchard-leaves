import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_SIZE = 96
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LetterboxResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized_image = image.resize(new_size, Image.BILINEAR)

        corners = [
            resized_image.getpixel((0, 0)),
            resized_image.getpixel((new_size[0] - 1, 0)),
            resized_image.getpixel((0, new_size[1] - 1)),
            resized_image.getpixel((new_size[0] - 1, new_size[1] - 1)),
        ]
        background_color = tuple(int(value) for value in np.median(corners, axis=0))

        canvas = Image.new("RGB", (self.size, self.size), background_color)
        offset = ((self.size - new_size[0]) // 2, (self.size - new_size[1]) // 2)
        canvas.paste(resized_image, offset)
        return canvas


class RandomRightAngleRotation:
    def __call__(self, image_tensor):
        quarter_turns = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(image_tensor, k=quarter_turns, dims=(1, 2))


BASE_TRANSFORM = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    RandomRightAngleRotation(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def build_preprocess_transform(image_size):
    return transforms.Compose([
        LetterboxResize(image_size),
        transforms.PILToTensor()
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


def cache_preprocessed_tensors(samples, cache_dir, image_size):
    preprocess_transform = build_preprocess_transform(image_size)
    cached_samples = []
    num_new_tensors = 0

    for image_path, label in samples:
        tensor_path = Path(cache_dir) / image_path.parent.name / f"{image_path.stem}.pt"

        if not tensor_path.exists():
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            image_tensor = preprocess_transform(load_rgb_image(image_path))
            temporary_path = tensor_path.with_suffix(".tmp")
            torch.save(image_tensor, temporary_path)
            temporary_path.replace(tensor_path)
            num_new_tensors += 1

        cached_samples.append((tensor_path, label))

    if num_new_tensors > 0:
        print(f"Cached {num_new_tensors} preprocessed tensors in {cache_dir}")

    return cached_samples


def balance_training_samples(samples, seed=RANDOM_SEED):
    random_generator = random.Random(seed)

    paths_by_label = {}
    for tensor_path, label in samples:
        paths_by_label.setdefault(label, []).append(tensor_path)

    target_samples = max(len(real_paths) for real_paths in paths_by_label.values())

    balanced_samples = []
    for label, real_paths in paths_by_label.items():
        balanced_samples += [(tensor_path, label) for tensor_path in real_paths]

        num_missing = target_samples - len(real_paths)
        for _ in range(num_missing):
            balanced_samples.append((random_generator.choice(real_paths), label))

    return balanced_samples


class OrchardDataset(Dataset):
    def __init__(self, samples, is_training=False):
        self.samples = samples
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tensor_path, label = self.samples[index]
        image_tensor = torch.load(tensor_path, weights_only=True)

        if self.is_training:
            return AUGMENT_TRANSFORM(image_tensor), label

        return BASE_TRANSFORM(image_tensor), label


def get_dataloaders(data_dir, processed_dir, batch_size=32, num_workers=2, seed=RANDOM_SEED, image_size=IMAGE_SIZE,
                    shuffle_seed=None):
    train_samples, validation_samples, test_samples, class_names = split_dataset_by_class(data_dir, seed=seed)

    cache_dir = Path(processed_dir) / f"letterbox_{image_size}px"
    train_samples = cache_preprocessed_tensors(train_samples, cache_dir, image_size)
    validation_samples = cache_preprocessed_tensors(validation_samples, cache_dir, image_size)
    test_samples = cache_preprocessed_tensors(test_samples, cache_dir, image_size)

    train_dataset = OrchardDataset(balance_training_samples(train_samples, seed=seed), is_training=True)
    validation_dataset = OrchardDataset(validation_samples, is_training=False)
    test_dataset = OrchardDataset(test_samples, is_training=False)

    if shuffle_seed is None:
        shuffle_seed = seed

    shuffle_generator = torch.Generator().manual_seed(shuffle_seed)
    keep_workers_alive = num_workers > 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              generator=shuffle_generator, persistent_workers=keep_workers_alive)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, persistent_workers=keep_workers_alive)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader, class_names


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "raw" / "leafscans-orchard-compressed"
    processed_dir = project_root / "data" / "processed"

    train_loader, validation_loader, test_loader, class_names = get_dataloaders(data_dir, processed_dir, batch_size=16)

    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {tuple(images.shape)}")
    print(f"Labels batch shape: {tuple(labels.shape)}")
