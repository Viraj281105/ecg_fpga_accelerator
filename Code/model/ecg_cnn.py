import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG_1DCNN(nn.Module):
    """
    Lightweight 1D CNN for ECG Arrhythmia Detection
    Optimized for FPGA implementation
    """
    
    def __init__(self, input_length=180, num_classes=2):
        super(ECG_1DCNN, self).__init__()
        
        # Conv Block 1: 1 -> 16 channels
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        
        # Conv Block 2: 16 -> 32 channels
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        
        # Conv Block 3: 32 -> 64 channels
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        
        # Calculate FC input size: 180 -> 90 -> 45 -> 22
        fc_input = 64 * 22
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input: (batch, 180) -> (batch, 1, 180)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = ECG_1DCNN()
    print(f"Model Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    dummy = torch.randn(4, 180)
    output = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {output.shape}")
    print("\n✅ Model architecture verified!")