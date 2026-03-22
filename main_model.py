# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        
        # Load pretrained ResNet-50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer (fc)
        # ResNet-50 outputs 2048-dim features after global average pooling
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Projection head: 2048 -> embedding_dim
        self.projection = nn.Linear(2048, embedding_dim)
    
    def forward(self, x):
        # Pass through ResNet backbone
        x = self.backbone(x)          # shape: (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)     # flatten -> (batch, 2048)
        x = self.projection(x)         # -> (batch, embedding_dim)
        x = F.normalize(x, p=2, dim=1) # L2 normalize
        return x


def save_model(model, path):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, embedding_dim=128, device='cpu'):
    """Load model weights from disk."""
    model = EmbeddingNet(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    # Quick test to verify the model works
    model = EmbeddingNet(embedding_dim=128)
    dummy_input = torch.randn(4, 3, 224, 224)  # batch of 4 images
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")       # should be (4, 128)
    print(f"Embedding norms: {output.norm(dim=1)}")  # should all be ~1.0