import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import Optional, Tuple


class VisionTransformer(nn.Module):
    """Fine-tuned ViT-B/16 model"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(VisionTransformer, self).__init__()
        
        # Load pretrained ViT-B/16
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.vit.embed_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.vit(x)  # [B, feature_dim]
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class EfficientNetV2(nn.Module):
    """EfficientNetV2-S baseline"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(EfficientNetV2, self).__init__()
        
        # Load pretrained EfficientNetV2-S
        self.backbone = timm.create_model(
            'efficientnetv2_s',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)  # [B, feature_dim]
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class HybridCNNTransformer(nn.Module):
    """Hybrid CNN + Transformer architecture"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(HybridCNNTransformer, self).__init__()
        
        # CNN feature extractor (ResNet50 backbone)
        self.cnn_backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0
        )
        
        # Get CNN feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            cnn_features = self.cnn_backbone(dummy_input)
            self.cnn_feature_dim = cnn_features.shape[1]
        
        # Project CNN features to transformer dimension
        self.feature_projection = nn.Linear(self.cnn_feature_dim, 512)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(512, max_len=196)  # 14x14 grid
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=512, num_heads=8, d_ff=2048, dropout=0.1)
            for _ in range(3)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.feature_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # [B, feature_dim]
        
        # For ResNet50, we need to get the spatial features before global pooling
        # Let's modify the approach to get spatial features
        # We'll use a custom forward to get the feature maps
        
        # Get spatial features from the last convolutional block
        x_features = x
        x_features = self.cnn_backbone.conv1(x_features)
        x_features = self.cnn_backbone.bn1(x_features)
        x_features = self.cnn_backbone.act1(x_features)
        x_features = self.cnn_backbone.maxpool(x_features)
        
        x_features = self.cnn_backbone.layer1(x_features)
        x_features = self.cnn_backbone.layer2(x_features)
        x_features = self.cnn_backbone.layer3(x_features)
        x_features = self.cnn_backbone.layer4(x_features)  # [B, 2048, 7, 7] for 224x224 input
        
        # Reshape to sequence
        b, c, h, w = x_features.shape
        x_features = x_features.view(b, c, h * w).transpose(1, 2)  # [B, seq_len, feature_dim]
        
        # Project to transformer dimension
        x_features = self.feature_projection(x_features)  # [B, seq_len, 512]
        
        # Add positional encoding
        x_features = self.pos_encoding(x_features)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x_features = transformer_block(x_features)
        
        # Global average pooling over sequence dimension
        x_features = x_features.mean(dim=1)  # [B, 512]
        
        # Classification
        logits = self.classifier(x_features)
        
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :]


def create_model(model_type: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create model based on type"""
    
    model_type = model_type.lower()
    
    if model_type == 'vit':
        return VisionTransformer(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'efficientnet':
        return EfficientNetV2(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'hybrid':
        return HybridCNNTransformer(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: 'vit', 'efficientnet', 'hybrid'")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    models = {
        'vit': VisionTransformer(),
        'efficientnet': EfficientNetV2(),
        'hybrid': HybridCNNTransformer()
    }
    
    for name, model in models.items():
        model = model.to(device)
        total_params, trainable_params = count_parameters(model)
        
        print(f"\n{name.upper()} Model:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
        
        print(f"✓ {name} model works correctly!")
