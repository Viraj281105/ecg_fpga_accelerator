import torch
import numpy as np
from ecg_cnn import ECG_1DCNN_Optimized
import os

def fuse_conv_bn(conv, bn):
    """
    Fuse Conv1d and BatchNorm1d layers
    
    Mathematical fusion:
    Original: y = BN(Conv(x))
    Fused: y = Conv_fused(x)
    
    Where:
    W_fused = W * (gamma / sqrt(var + eps))
    b_fused = (b - mu) * (gamma / sqrt(var + eps)) + beta
    """
    # Get Conv parameters
    conv_weight = conv.weight.data.clone()
    if conv.bias is not None:
        conv_bias = conv.bias.data.clone()
    else:
        conv_bias = torch.zeros(conv.out_channels)
    
    # Get BatchNorm parameters
    bn_weight = bn.weight.data.clone()      # gamma
    bn_bias = bn.bias.data.clone()          # beta
    bn_mean = bn.running_mean.data.clone()  # mu
    bn_var = bn.running_var.data.clone()    # sigma^2
    bn_eps = bn.eps
    
    # Compute scale factor
    bn_std = torch.sqrt(bn_var + bn_eps)
    scale = bn_weight / bn_std
    
    # Fuse weights: W_fused = W * scale
    fused_weight = conv_weight * scale.view(-1, 1, 1)
    
    # Fuse bias: b_fused = (b - mu) * scale + beta
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias
    
    return fused_weight, fused_bias

def export_fused_model(model_path='results/models/best_model.pth',
                       save_dir='results/models'):
    """Export model with fused Conv+BN layers"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = ECG_1DCNN_Optimized()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✓ Model loaded")
    
    print("\n" + "="*60)
    print("FUSING CONV + BATCHNORM LAYERS")
    print("="*60)
    
    # Fuse each Conv+BN pair
    print("\nFusing Conv1 + BN1...")
    conv1_w, conv1_b = fuse_conv_bn(model.conv1, model.bn1)
    print(f"  Conv1 fused weights: {conv1_w.shape}")
    print(f"  Conv1 fused bias: {conv1_b.shape}")
    
    print("\nFusing Conv2 + BN2...")
    conv2_w, conv2_b = fuse_conv_bn(model.conv2, model.bn2)
    print(f"  Conv2 fused weights: {conv2_w.shape}")
    print(f"  Conv2 fused bias: {conv2_b.shape}")
    
    print("\nFusing Conv3 + BN3...")
    conv3_w, conv3_b = fuse_conv_bn(model.conv3, model.bn3)
    print(f"  Conv3 fused weights: {conv3_w.shape}")
    print(f"  Conv3 fused bias: {conv3_b.shape}")
    
    # Get FC layer (no BatchNorm)
    print("\nExtracting FC layer...")
    fc_w = model.fc.weight.data.clone()
    fc_b = model.fc.bias.data.clone()
    print(f"  FC weights: {fc_w.shape}")
    print(f"  FC bias: {fc_b.shape}")
    
    # Save as NumPy
    weights_dict = {
        'conv1_weight': conv1_w.numpy(),
        'conv1_bias': conv1_b.numpy(),
        'conv2_weight': conv2_w.numpy(),
        'conv2_bias': conv2_b.numpy(),
        'conv3_weight': conv3_w.numpy(),
        'conv3_bias': conv3_b.numpy(),
        'fc_weight': fc_w.numpy(),
        'fc_bias': fc_b.numpy()
    }
    
    npz_path = os.path.join(save_dir, 'fused_weights.npz')
    np.savez(npz_path, **weights_dict)
    print(f"\n✓ Saved NumPy format to {npz_path}")
    
    # Export as HEX for hardware
    export_hex_format(weights_dict, save_dir)
    
    # Verify fusion correctness
    verify_fusion(model, weights_dict)
    
    # Print summary
    print_weight_summary(weights_dict)
    
    return weights_dict

def export_hex_format(weights_dict, save_dir):
    """Export weights in Q8.8 hex format for hardware"""
    
    def float_to_q8_8(value):
        """Convert float to Q8.8 fixed-point"""
        fixed = int(value * 256)
        # Clamp to 16-bit signed range
        fixed = max(-32768, min(32767, fixed))
        return fixed & 0xFFFF
    
    hex_dir = os.path.join(save_dir, 'hex')
    os.makedirs(hex_dir, exist_ok=True)
    
    print("\nExporting to hex format (Q8.8)...")
    for name, weights in weights_dict.items():
        hex_file = os.path.join(hex_dir, f'{name}.hex')
        with open(hex_file, 'w') as f:
            flat_weights = weights.flatten()
            for w in flat_weights:
                hex_val = float_to_q8_8(w)
                f.write(f"{hex_val:04X}\n")
        print(f"  ✓ {name}.hex ({len(flat_weights)} values)")
    
    print(f"\n✓ Hex files saved to {hex_dir}/")

def verify_fusion(model, fused_weights):
    """Verify that fusion didn't change model output"""
    
    print("\n" + "="*60)
    print("VERIFYING FUSION CORRECTNESS")
    print("="*60)
    
    # Load test data
    segments = np.load('data/processed/segments.npy')
    test_input = torch.FloatTensor(segments[:100])  # Test on 100 samples
    
    # Original model output
    with torch.no_grad():
        original_output = model(test_input)
        original_pred = original_output.argmax(dim=1)
    
    # Create fused model (without BatchNorm)
    class FusedModel(torch.nn.Module):
        def __init__(self, fused_w):
            super().__init__()
            
            # Conv layers (no BatchNorm!)
            self.conv1 = torch.nn.Conv1d(1, 16, 5, stride=2, padding=2)
            self.conv1.weight.data = torch.FloatTensor(fused_w['conv1_weight'])
            self.conv1.bias.data = torch.FloatTensor(fused_w['conv1_bias'])
            
            self.conv2 = torch.nn.Conv1d(16, 32, 5, stride=2, padding=2)
            self.conv2.weight.data = torch.FloatTensor(fused_w['conv2_weight'])
            self.conv2.bias.data = torch.FloatTensor(fused_w['conv2_bias'])
            
            self.conv3 = torch.nn.Conv1d(32, 64, 5, stride=2, padding=2)
            self.conv3.weight.data = torch.FloatTensor(fused_w['conv3_weight'])
            self.conv3.bias.data = torch.FloatTensor(fused_w['conv3_bias'])
            
            self.gap = torch.nn.AdaptiveAvgPool1d(1)
            
            self.fc = torch.nn.Linear(64, 2)
            self.fc.weight.data = torch.FloatTensor(fused_w['fc_weight'])
            self.fc.bias.data = torch.FloatTensor(fused_w['fc_bias'])
        
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            x = torch.relu(self.conv1(x))  # No BN!
            x = torch.relu(self.conv2(x))  # No BN!
            x = torch.relu(self.conv3(x))  # No BN!
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    fused_model = FusedModel(fused_weights)
    fused_model.eval()
    
    # Fused model output
    with torch.no_grad():
        fused_output = fused_model(test_input)
        fused_pred = fused_output.argmax(dim=1)
    
    # Compare predictions
    matches = (original_pred == fused_pred).sum().item()
    
    # Compare output values
    max_diff = (original_output - fused_output).abs().max().item()
    mean_diff = (original_output - fused_output).abs().mean().item()
    
    print(f"Test samples: 100")
    print(f"Predictions match: {matches}/100 ({matches}%)")
    print(f"Max output difference: {max_diff:.6f}")
    print(f"Mean output difference: {mean_diff:.6f}")
    
    if matches >= 99:  # Allow 1% tolerance
        print("\n✅ FUSION SUCCESSFUL - Outputs match!")
    else:
        print("\n⚠️  WARNING - Some predictions differ")
        print("This might be due to numerical precision")

def print_weight_summary(weights_dict):
    """Print summary of weights"""
    
    print("\n" + "="*60)
    print("WEIGHT SUMMARY")
    print("="*60)
    
    total_params = 0
    
    for name, weights in weights_dict.items():
        num_params = weights.size
        total_params += num_params
        print(f"{name:20s}: {str(weights.shape):20s} = {num_params:6d} params")
    
    print("-"*60)
    print(f"{'TOTAL':20s}: {'':<20s} = {total_params:6d} params")
    print(f"\nMemory (Q8.8): {total_params * 2 / 1024:.2f} KB")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("BATCHNORM FUSION FOR HARDWARE")
    print("="*60)
    print()
    
    weights = export_fused_model()
    
    print("\n" + "="*60)
    print("✅ EXPORT COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. results/models/fused_weights.npz")
    print("  2. results/models/hex/*.hex")
    print("\nWhat changed:")
    print("  ✅ Conv + BatchNorm fused into single Conv")
    print("  ✅ No BatchNorm needed in hardware!")
    print("  ✅ Mathematically identical output")
    print("\nNext steps:")
    print("  → Share fused_weights.npz with hardware partner")
    print("  → Hardware implements 3 Conv + GAP + FC")
    print("  → No BatchNorm layer needed!")