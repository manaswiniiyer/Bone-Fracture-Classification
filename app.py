import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import yaml
import os
from pathlib import Path
import tempfile
import traceback
import pandas as pd

from data_loader import get_transforms
from model import create_model
from utils import load_checkpoint, get_device
from gradcam import ModelGradCAM


class BoneFractureApp:
    """Gradio web interface for bone fracture classification"""
    
    def __init__(self, config_path='config.yaml', checkpoint_dir='checkpoints'):
        self.config_path = config_path
        self.checkpoint_dir = checkpoint_dir
        self.device = get_device()
        self.model = None
        self.class_names = None
        self.config = None
        self.grad_cam = None
        
        # Load configuration and model
        self.load_model()
    
    def load_model(self):
        """Load model and configuration"""
        
        try:
            # Load configuration
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.class_names = self.config['classes']
            
            # Create model
            model_type = self.config['model']['type']
            num_classes = self.config['model']['num_classes']
            
            self.model = create_model(model_type, num_classes, pretrained=False)
            self.model = self.model.to(self.device)
            
            # Load best checkpoint
            best_checkpoint = os.path.join(self.checkpoint_dir, 'best_model.pth')
            
            if not os.path.exists(best_checkpoint):
                # Try to find any checkpoint
                checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                                  if f.endswith('.pth') and 'best' in f]
                if checkpoint_files:
                    best_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[0])
                else:
                    raise FileNotFoundError("No checkpoint found. Please train the model first.")
            
            load_checkpoint(best_checkpoint, self.model, None, None, self.device)
            
            # Create Grad-CAM
            self.grad_cam = ModelGradCAM.create_grad_cam(model_type, self.model)
            
            print(f"Model loaded successfully: {model_type}")
            print(f"Checkpoint: {best_checkpoint}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating demo model for testing...")
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create a demo model for testing when no checkpoint is available"""
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['classes']
        
        # Create untrained model
        model_type = self.config['model']['type']
        num_classes = self.config['model']['num_classes']
        
        self.model = create_model(model_type, num_classes, pretrained=False)
        self.model = self.model.to(self.device)
        
        # Create Grad-CAM
        try:
            self.grad_cam = ModelGradCAM.create_grad_cam(model_type, self.model)
        except:
            self.grad_cam = None
        
        print("Demo model created (untrained)")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""

        if image is None:
            return None

        # Convert PIL → numpy
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image_np = np.array(image)
        else:
            image_np = image

        # Ensure correct dtype
        image_np = image_np.astype(np.uint8)

        # Get transforms
        transform = get_transforms(self.config, is_train=False)

        # Apply transform correctly
        transformed = transform(image=image_np)

        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        return image_tensor
    
    def predict(self, image):
        """Make prediction on uploaded image"""
        
        if image is None:
            return "Please upload an X-ray image", {}, None
        
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item() * 100
            
            # Create confidence dictionary
            confidence_values = probabilities[0].detach().cpu().numpy() * 100

            confidence_data = pd.DataFrame({
                "class": self.class_names,
                "confidence": confidence_values
            })
            # Generate Grad-CAM
            try:
                if self.grad_cam is not None:

                    cam, _ = self.grad_cam.generate_cam(input_tensor)

                    if isinstance(image, Image.Image):
                        img_array = np.array(image)
                    else:
                        img_array = image

                    cam_resized = cv2.resize(cam.squeeze(), (img_array.shape[1], img_array.shape[0]))

                    cam_resized = cam_resized - cam_resized.min()
                    cam_resized = cam_resized / (cam_resized.max() + 1e-8)

                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    overlay = cv2.addWeighted(img_array.astype(np.uint8), 0.6, heatmap, 0.4, 0)

                    gradcam_image = Image.fromarray(overlay)

                else:
                    gradcam_image = None

            except Exception as e:
                print("GradCAM error:", e)
                gradcam_image = None
                            
            result_text = f"**Predicted Fracture Type:** {predicted_class}\n**Confidence:** {confidence_score:.2f}%"
            
            #return result_text, confidences, gradcam_image
            return result_text, confidence_data, gradcam_image
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(f"Prediction error: {traceback.format_exc()}")
            return error_msg, {}, None
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        """
        
        with gr.Blocks(css=css, title=self.config.get('app', {}).get('title', 'Bone Fracture Classification')) as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🦴 Bone Fracture Classification System</h1>
                <p>Upload an X-ray image to classify the type of fracture using advanced AI models</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 📸 Upload X-ray Image")
                    input_image = gr.Image(
                        label="X-ray Image",
                        type="pil",
                        height=400
                    )
                    
                    # Model info
                    model_info = gr.Markdown(f"""
                    **Model Information:**
                    - Architecture: {self.config['model']['type'].upper()}
                    - Classes: {len(self.class_names)} fracture types
                    - Device: {str(self.device).upper()}
                    """)
                    
                    predict_btn = gr.Button(
                        "🔍 Analyze Fracture",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Results section
                    gr.Markdown("### 📊 Analysis Results")
                    
                    prediction_text = gr.Markdown(
                        "Upload an image and click 'Analyze Fracture' to see results",
                        elem_classes=["prediction-box"]
                    )
                    
                    # Confidence scores
                    with gr.Row():
                        confidence_plot = gr.BarPlot(
                    #       label="Confidence Scores by Class",
                            title="Confidence Scores",
                            x="class",
                            y="confidence",
                            height=300,
                    #        show_label=True
                        )
                    
                    # Grad-CAM visualization
                    gradcam_output = gr.Image(
                        label="Grad-CAM Heatmap (Attention Visualization)",
                        type="pil",
                        height=400
                    )
            
            # Examples section
            gr.Markdown("### 💡 Example Usage")
            gr.Markdown("""
            1. Upload an X-ray image showing a bone fracture
            2. Click "Analyze Fracture" to process the image
            3. View the predicted fracture type and confidence scores
            4. Examine the Grad-CAM heatmap to see which regions the model focused on
            """)
            
            # Class information
            with gr.Accordion("📚 Fracture Types Information", open=False):
                class_info = gr.Markdown(self.get_class_information())
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f0f0f0; border-radius: 10px;">
                <p><strong>Disclaimer:</strong> This system is for educational and research purposes only. 
                Always consult with qualified medical professionals for diagnosis and treatment.</p>
                <p>Powered by PyTorch • Vision Transformers • EfficientNet • Hybrid CNN-Transformer</p>
            </div>
            """)
            
            # Event handlers
            predict_btn.click(
                fn=self.predict,
                inputs=[input_image],
                outputs=[prediction_text, confidence_plot, gradcam_output]
            )
            
            # Also allow prediction on image upload
            input_image.change(
                fn=self.predict,
                inputs=[input_image],
                outputs=[prediction_text, confidence_plot, gradcam_output]
            )
        
        return interface
    
    def get_class_information(self):
        """Get information about fracture types"""
        
        class_descriptions = {
            "simple": "A clean break with little damage to surrounding tissue",
            "comminuted": "Bone shattered into three or more pieces",
            "spiral": "Fracture that spirals around the bone, caused by twisting",
            "stress": "Small crack in bone caused by repetitive force",
            "greenstick": "Fracture where bone bends and cracks but doesn't break completely",
            "compound": "Fracture where bone pierces through the skin",
            "pathological": "Fracture caused by disease weakening the bone"
        }
        
        info = "**Fracture Classifications:**\n\n"
        for class_name in self.class_names:
            description = class_descriptions.get(class_name, "No description available")
            info += f"**{class_name.replace('_', ' ').title()}:** {description}\n\n"
        
        return info
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        
        interface = self.create_interface()
        
        # Default launch parameters
        launch_kwargs = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'debug': False,
            'show_error': True,
        }
        
        # Override with user-provided kwargs
        launch_kwargs.update(kwargs)
        
        print("🚀 Starting Bone Fracture Classification Web App...")
        print(f"📍 Server will be available at: http://{launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        
        interface.launch(**launch_kwargs)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch bone fracture classification web app')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--share', action='store_true', help='Create public shareable link')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode without trained model')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    
    args = parser.parse_args()
    
    # Enable demo mode if specified
    if args.demo:
        print("🎭 Running in DEMO mode - using untrained model")
        # Create dummy checkpoint directory if it doesn't exist
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create and launch app
    try:
        app = BoneFractureApp(args.config, args.checkpoint_dir)
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print(f"📋 Full error: {traceback.format_exc()}")
        
        # Try to launch in demo mode as fallback
        if not args.demo:
            print("\n🔄 Attempting to launch in demo mode...")
            try:
                app = BoneFractureApp(args.config, args.checkpoint_dir)
                app.launch(
                    server_name=args.host,
                    server_port=args.port,
                    share=args.share
                )
            except Exception as demo_error:
                print(f"❌ Demo mode also failed: {demo_error}")


if __name__ == "__main__":
    main()
