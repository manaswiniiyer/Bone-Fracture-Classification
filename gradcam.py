import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_loader import get_transforms
from model import create_model
from utils import load_checkpoint, get_device


class GradCAM:
    """Grad-CAM implementation for visualizing model attention"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        """Register hooks to capture gradients and activations"""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found")
        print("GradCAM attached to layer:", self.target_layer_name)
        target_layer.register_forward_hook(forward_hook)
        #target_layer.register_backward_hook(backward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate class activation map"""
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # [H, W]
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalization
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        #return cam.cpu().numpy(), class_idx
        return cam.detach().cpu().numpy(), class_idx
    
    def visualize_cam(self, image_path, cam, class_name, save_path=None):
        """Visualize CAM overlaid on original image"""
        
        # Load original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(original_image)
        ax1.set_title('Original X-ray')
        ax1.axis('off')
        
        ax2.imshow(cam_resized, cmap='jet')
        ax2.set_title(f'Grad-CAM\n({class_name})')
        ax2.axis('off')
        
        ax3.imshow(overlay)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return overlay


class ModelGradCAM:
    """Model-specific Grad-CAM implementations"""
    
    @staticmethod
    def get_target_layer(model_type, model):

        if model_type.lower() == "vit":
            # last transformer block normalization
            return "vit.blocks.11.norm1"

        elif model_type.lower() == "efficientnet":
            # last conv layer of EfficientNetV2
            return "backbone.conv_head"

        elif model_type.lower() == "hybrid":
            # last CNN layer before transformer
            return "cnn_backbone.layer4.2.conv3"

        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def create_grad_cam(model_type, model):
        """Create Grad-CAM instance for specific model"""
        
        target_layer = ModelGradCAM.get_target_layer(model_type, model)
        return GradCAM(model, target_layer)


def preprocess_image(image_path, config, device):
    """Preprocess image for model input"""
    
    transform = get_transforms(config, is_train=False)
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # Apply transforms
    transformed = transform(image=image_array)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    return image_tensor, image_array


def generate_gradcam_visualization(model, image_path, class_names, config, device, save_dir=None):
    """Generate Grad-CAM visualization for a single image"""
    
    # Create Grad-CAM
    grad_cam = ModelGradCAM.create_grad_cam(config['model']['type'], model)
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path, config, device)
    
    # Generate CAM
    cam, predicted_class_idx = grad_cam.generate_cam(input_tensor)
    predicted_class = class_names[predicted_class_idx]
    
    # Create save path
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        image_name = Path(image_path).stem
        save_path = save_dir / f'{image_name}_gradcam.png'
    else:
        save_path = None
    
    # Visualize
    overlay = grad_cam.visualize_cam(image_path, cam, predicted_class, save_path)
    
    return cam, predicted_class, overlay


def batch_gradcam_generation(model, image_paths, class_names, config, device, save_dir):
    """Generate Grad-CAM for multiple images"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        try:
            cam, predicted_class, overlay = generate_gradcam_visualization(
                model, image_path, class_names, config, device, save_dir
            )
            
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'cam': cam,
                'overlay': overlay
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    return results


def demo_gradcam():
    """Demo Grad-CAM functionality"""
    
    # Create dummy model and image
    device = get_device()
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config['model']['type'], config['model']['num_classes'], pretrained=False)
    model = model.to(device)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_path = 'dummy_image.jpg'
    cv2.imwrite(dummy_path, dummy_image)
    
    # Generate Grad-CAM
    try:
        grad_cam = ModelGradCAM.create_grad_cam(config['model']['type'], model)
        input_tensor, _ = preprocess_image(dummy_path, config, device)
        
        cam, class_idx = grad_cam.generate_cam(input_tensor)
        print(f"Grad-CAM generated successfully! Shape: {cam.shape}")
        print(f"Predicted class index: {class_idx}")
        
        # Clean up
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            
    except Exception as e:
        print(f"Error in Grad-CAM demo: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output', type=str, default='gradcam_results', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    args = parser.parse_args()
    
    if args.demo:
        demo_gradcam()
        return
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = get_device()
    
    # Create model
    model = create_model(config['model']['type'], config['model']['num_classes'], pretrained=False)
    model = model.to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, None, None, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Class names
    class_names = config['classes']
    
    # Process images
    if args.image:
        # Single image
        print(f"Processing single image: {args.image}")
        cam, predicted_class, overlay = generate_gradcam_visualization(
            model, args.image, class_names, config, device, args.output
        )
        print(f"Predicted class: {predicted_class}")
        
    elif args.image_dir:
        # Directory of images
        import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
        
        print(f"Found {len(image_paths)} images")
        
        results = batch_gradcam_generation(
            model, image_paths, class_names, config, device, args.output
        )
        
        print(f"Generated Grad-CAM for {len(results)} images")
        print(f"Results saved to: {args.output}")
        
    else:
        print("Please provide either --image or --image_dir")


if __name__ == "__main__":
    import os
    main()
