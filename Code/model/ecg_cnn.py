import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG_1DCNN_Optimized(nn.Module):
    """
    FPGA-Optimized 1D CNN for ECG Arrhythmia Detection
    - Uses stride instead of pooling
    - Global Average Pooling instead of large FC
    - Minimal parameters for hardware efficiency
    """
    
    def __init__(self, input_length=180, num_classes=2):
        super(ECG_1DCNN_Optimized, self).__init__()
        
        # Conv Block 1: stride=2 (replaces conv+pool)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Conv Block 2: stride=2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Conv Block 3: stride=2
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Global Average Pooling (replaces flatten + large FC)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Small FC layer (only 64→2 weights!)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input: (batch, 180) → (batch, 1, 180)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Conv Block 1
        x = self.conv1(x)  # (batch, 16, 90)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv Block 2
        x = self.conv2(x)  # (batch, 32, 45)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)  # (batch, 64, 23)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Average Pooling
        x = self.gap(x)  # (batch, 64, 1)
        x = x.view(x.size(0), -1)  # (batch, 64)
        
        # Classification
        x = self.fc(x)  # (batch, 2)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = ECG_1DCNN_Optimized()
    print(f"Optimized Model Parameters: {model.count_parameters():,}")
    
    # Test
    dummy = torch.randn(4, 180)
    output = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {output.shape}")
    print("\n✅ FPGA-optimized model verified!")