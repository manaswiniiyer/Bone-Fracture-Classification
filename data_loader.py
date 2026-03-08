import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from collections import Counter
import glob


from albumentations.core.transforms_interface import ImageOnlyTransform

class CLAHE(ImageOnlyTransform):
    """Custom CLAHE transform compatible with Albumentations"""

    def __init__(self, clip_limit=2.0, tile_grid_size=8, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image, **params):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.tile_grid_size, self.tile_grid_size)
            )

            enhanced = clahe.apply(gray)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

            return enhanced

        return image

class XRayDataset(Dataset):
    """X-ray dataset with custom augmentations"""
    
    def __init__(self, image_paths, labels, transform=None, is_grayscale=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_grayscale = is_grayscale
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


def get_transforms(config, is_train=True):
    """Get data transforms for training or validation"""
    
    # Base transforms
    normalize = A.Normalize(
        mean=config['augmentation']['normalize_mean'],
        std=config['augmentation']['normalize_std']
    )
    
    if is_train:
        transform = A.Compose([
            A.Resize(config['data']['image_size'], config['data']['image_size']),
            A.HorizontalFlip(p=config['augmentation']['horizontal_flip_prob']),
            A.Rotate(limit=config['augmentation']['rotation_degrees'], p=0.5),
            CLAHE(
                clip_limit=config['augmentation']['clahe_clip_limit'],
                tile_grid_size=config['augmentation']['clahe_tile_grid_size']
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            normalize,
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(config['data']['image_size'], config['data']['image_size']),
            CLAHE(
                clip_limit=config['augmentation']['clahe_clip_limit'],
                tile_grid_size=config['augmentation']['clahe_tile_grid_size']
            ),
            normalize,
            ToTensorV2()
        ])
    
    return transform


def load_dataset_from_folder(data_dir, demo_mode=False):
    """Load dataset from ImageFolder format"""
    
    if demo_mode:
        # Create dummy dataset for demo
        print("Creating dummy dataset for demo mode...")
        image_paths = []
        labels = []
        class_names = ['simple', 'comminuted', 'spiral', 'stress', 'greenstick', 'compound', 'pathological']
        
        for i, class_name in enumerate(class_names):
            # Create dummy paths (these won't actually exist, but will trigger demo mode)
            for j in range(10):  # 10 dummy images per class
                image_paths.append(f"dummy_{class_name}_{j}.jpg")
                labels.append(i)
        
        return image_paths, labels, class_names
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    labels = []
    
    # Get class folders
    class_folders = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
    class_folders.sort()
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    for class_idx, class_name in enumerate(class_folders):
        class_dir = os.path.join(data_dir, class_name)
        
        # Find all images in this class folder
        class_images = []
        for ext in image_extensions:
            class_images.extend(glob.glob(os.path.join(class_dir, ext)))
            class_images.extend(glob.glob(os.path.join(class_dir, ext.upper())))
        
        print(f"Class {class_name}: {len(class_images)} images")
        
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
    
    if len(image_paths) == 0:
        raise ValueError("No images found in the dataset directory")
    
    return image_paths, labels, class_folders


def create_stratified_split(image_paths, labels, train_split=0.7, val_split=0.1, test_split=0.2, random_seed=42):
    """Create stratified train/val/test split"""
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, 
        test_size=test_split, 
        stratify=labels, 
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_size_adjusted = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        stratify=y_temp, 
        random_state=random_seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_loaders(config):
    """Create data loaders for training, validation, and testing"""
    
    # Load dataset
    data_dir = config['data']['data_dir']
    demo_mode = config.get('app', {}).get('demo_mode', False)
    
    image_paths, labels, class_names = load_dataset_from_folder(data_dir, demo_mode)
    
    # Create stratified split
    X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_split(
        image_paths, labels,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )
    
    print(f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = XRayDataset(
        X_train, y_train, 
        transform=get_transforms(config, is_train=True)
    )
    
    val_dataset = XRayDataset(
        X_val, y_val, 
        transform=get_transforms(config, is_train=False)
    )
    
    test_dataset = XRayDataset(
        X_test, y_test, 
        transform=get_transforms(config, is_train=False)
    )
    
    # Calculate class weights for balanced sampling
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names, class_weights


def get_demo_data_loader(config):
    """Create a demo data loader with dummy data"""
    
    # Create dummy dataset
    dummy_images = torch.randn(10, 3, config['data']['image_size'], config['data']['image_size'])
    dummy_labels = torch.randint(0, config['model']['num_classes'], (10,))
    
    dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    return loader, config['classes'], {i: 1.0 for i in range(config['model']['num_classes'])}


if __name__ == "__main__":
    # Test the data loader
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        train_loader, val_loader, test_loader, class_names, class_weights = get_data_loaders(config)
        print("Data loaders created successfully!")
        print(f"Class names: {class_names}")
        print(f"Class weights: {class_weights}")
        
        # Test a batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        print("Trying demo mode...")
        config['app']['demo_mode'] = True
        train_loader, class_names, class_weights = get_demo_data_loader(config)
        print("Demo data loader created successfully!")
