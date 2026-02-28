import torch
import numpy as np
from ecg_cnn import ECG_1DCNN_Optimized

def float_to_q8_8(value):
    """Convert float to Q8.8"""
    fixed = int(value * 256)
    return max(-32768, min(32767, fixed))

def q8_8_to_float(fixed):
    """Convert Q8.8 to float"""
    return fixed / 256.0

def generate_layer_by_layer_outputs():
    """Generate intermediate outputs for debugging"""
    
    # Load model
    model = ECG_1DCNN_Optimized()
    model.load_state_dict(torch.load('results/models/best_model.pth', map_location='cpu'))
    model.eval()
    
    # Load fused weights
    fused = np.load('results/models/fused_weights.npz')
    
    print("="*60)
    print("LAYER-BY-LAYER DEBUG OUTPUTS")
    print("="*60)
    
    # Test cases
    test_cases = [
        ("All zeros", np.zeros(180)),
        ("All ones", np.ones(180)),
        ("Simple ramp", np.arange(180) / 180.0),
    ]
    
    for name, input_data in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        
        # Convert to Q8.8
        input_q8 = np.array([float_to_q8_8(x) for x in input_data])
        
        print(f"\nInput (first 5 samples):")
        for i in range(5):
            print(f"  [{i}] Float: {input_data[i]:8.4f} | Q8.8: {input_q8[i]:6d} | Hex: 0x{input_q8[i]&0xFFFF:04X}")
        
        # Python inference
        with torch.no_grad():
            x = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0)
            
            # Conv1
            x1 = model.conv1(x)
            x1 = model.bn1(x1)
            x1_relu = torch.relu(x1)
            
            print(f"\nAfter Conv1+BN+ReLU:")
            print(f"  Shape: {x1_relu.shape}")
            print(f"  First 5 values: {x1_relu[0, :5, 0].numpy()}")
            
            # Conv2
            x2 = model.conv2(x1_relu)
            x2 = model.bn2(x2)
            x2_relu = torch.relu(x2)
            
            print(f"\nAfter Conv2+BN+ReLU:")
            print(f"  Shape: {x2_relu.shape}")
            print(f"  First 5 values: {x2_relu[0, :5, 0].numpy()}")
            
            # Conv3
            x3 = model.conv3(x2_relu)
            x3 = model.bn3(x3)
            x3_relu = torch.relu(x3)
            
            print(f"\nAfter Conv3+BN+ReLU:")
            print(f"  Shape: {x3_relu.shape}")
            print(f"  First 5 values: {x3_relu[0, :5, 0].numpy()}")
            
            # GAP
            gap = model.gap(x3_relu)
            gap_flat = gap.view(-1)
            
            print(f"\nAfter GAP:")
            print(f"  Shape: {gap_flat.shape}")
            print(f"  First 5 values: {gap_flat[:5].numpy()}")
            
            # FC
            fc_out = model.fc(gap_flat.unsqueeze(0))
            
            print(f"\nFinal Output (FC):")
            print(f"  Values: {fc_out[0].numpy()}")
            print(f"  Prediction: {fc_out.argmax().item()}")
    
    # Generate expected outputs for Conv1 with simple inputs
    print(f"\n{'='*60}")
    print("CONV1 MANUAL CALCULATION (for verification)")
    print(f"{'='*60}")
    
    # Simple test: all ones input
    input_ones = np.ones(180)
    conv1_weight = fused['conv1_weight']
    conv1_bias = fused['conv1_bias']
    
    print(f"\nInput: All ones (180 samples)")
    print(f"Conv1 weight shape: {conv1_weight.shape}")
    print(f"Conv1 has {conv1_weight.shape[0]} output channels")
    
    # Manual convolution for first output
    print(f"\nManual calculation for output channel 0, position 0:")
    kernel = conv1_weight[0, 0, :]  # First output channel, first input channel, all kernel positions
    
    # Stride=2, so we take samples [0,1,2,3,4] for first output
    input_window = input_ones[0:5]
    
    print(f"  Kernel: {kernel}")
    print(f"  Input window: {input_window}")
    print(f"  Dot product: {np.dot(kernel, input_window)}")
    print(f"  Add bias: {np.dot(kernel, input_window) + conv1_bias[0]}")
    print(f"  After ReLU: {max(0, np.dot(kernel, input_window) + conv1_bias[0])}")
    
    # In Q8.8
    kernel_q8 = [float_to_q8_8(k) for k in kernel]
    input_q8 = [float_to_q8_8(i) for i in input_window]
    bias_q8 = float_to_q8_8(conv1_bias[0])
    
    print(f"\nIn Q8.8 fixed-point:")
    print(f"  Kernel Q8.8: {kernel_q8}")
    print(f"  Input Q8.8: {input_q8}")
    
    # MAC operation
    accumulator = 0
    for k, i in zip(kernel_q8, input_q8):
        product = k * i  # Q16.16
        accumulator += product
    
    print(f"  Accumulator (Q16.16): {accumulator}")
    print(f"  After >>8 (Q8.8): {accumulator >> 8}")
    print(f"  Add bias Q8.8: {(accumulator >> 8) + bias_q8}")
    print(f"  After ReLU: {max(0, (accumulator >> 8) + bias_q8)}")
    print(f"  Back to float: {q8_8_to_float(max(0, (accumulator >> 8) + bias_q8))}")

if __name__ == "__main__":
    generate_layer_by_layer_outputs()