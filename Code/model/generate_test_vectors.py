import torch
import numpy as np
from ecg_cnn import ECG_1DCNN_Optimized
import os

class TestVectorGenerator:
    """Generate test vectors for RTL verification"""
    
    def __init__(self, model_path='results/models/best_model.pth'):
        self.model = ECG_1DCNN_Optimized()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.model.cpu()
        
    def float_to_fixed(self, value, int_bits=8, frac_bits=8):
        """Convert float to fixed-point representation (Q8.8 format)"""
        scale = 2 ** frac_bits
        fixed_value = int(value * scale)
        
        # Clamp to representable range
        max_val = (2 ** (int_bits + frac_bits - 1)) - 1
        min_val = -(2 ** (int_bits + frac_bits - 1))
        fixed_value = max(min_val, min(max_val, fixed_value))
        
        return fixed_value
    
    def fixed_to_float(self, fixed_value, frac_bits=8):
        """Convert fixed-point back to float"""
        scale = 2 ** frac_bits
        return fixed_value / scale
    
    def generate_test_cases(self, num_cases=50):
        """Generate diverse test cases"""
        print("Generating test cases...")
        
        # Load processed data
        segments = np.load('data/processed/segments.npy')
        labels = np.load('data/processed/labels.npy')
        
        # Select diverse test cases
        np.random.seed(42)  # Reproducibility
        
        # Get indices for each class
        normal_idx = np.where(labels == 0)[0]
        arrhythmia_idx = np.where(labels == 1)[0]
        
        # Balance classes
        num_per_class = num_cases // 2
        selected_normal = np.random.choice(normal_idx, num_per_class, replace=False)
        selected_arrhythmia = np.random.choice(arrhythmia_idx, num_per_class, replace=False)
        
        selected_idx = np.concatenate([selected_normal, selected_arrhythmia])
        np.random.shuffle(selected_idx)
        
        test_segments = segments[selected_idx]
        test_labels = labels[selected_idx]
        
        print(f"✓ Generated {len(test_segments)} test cases")
        print(f"  Normal: {np.sum(test_labels == 0)}")
        print(f"  Arrhythmia: {np.sum(test_labels == 1)}")
        
        return test_segments, test_labels
    
    def generate_vectors(self, num_cases=50, save_dir='data/test_vectors'):
        """Generate test vectors in multiple formats"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate test cases
        test_segments, test_labels = self.generate_test_cases(num_cases)
        
        # Run inference
        print("\nRunning inference...")
        with torch.no_grad():
            inputs = torch.FloatTensor(test_segments)
            outputs = self.model(inputs)
            predictions = outputs.argmax(dim=1).numpy()
        
        accuracy = np.mean(predictions == test_labels) * 100
        print(f"✓ Test accuracy: {accuracy:.2f}%")
        
        # Save in different formats
        self.save_numpy_format(test_segments, test_labels, predictions, save_dir)
        self.save_text_format(test_segments, test_labels, predictions, save_dir)
        self.save_hex_format(test_segments, test_labels, predictions, save_dir)
        self.save_verilog_format(test_segments, test_labels, predictions, save_dir)
        
        print(f"\n✅ Test vectors saved to {save_dir}/")
        
        return test_segments, test_labels, predictions
    
    def save_numpy_format(self, segments, labels, predictions, save_dir):
        """Save as NumPy arrays"""
        np.savez(
            os.path.join(save_dir, 'test_vectors.npz'),
            inputs=segments,
            labels=labels,
            predictions=predictions
        )
        print("✓ Saved NumPy format")
    
    def save_text_format(self, segments, labels, predictions, save_dir):
        """Save as human-readable text"""
        with open(os.path.join(save_dir, 'test_vectors.txt'), 'w') as f:
            f.write("ECG Arrhythmia Test Vectors\n")
            f.write("="*60 + "\n")
            f.write(f"Total cases: {len(segments)}\n")
            f.write(f"Input size: 180 samples per case\n")
            f.write(f"Output: Binary classification (0=Normal, 1=Arrhythmia)\n")
            f.write("="*60 + "\n\n")
            
            for i in range(len(segments)):
                f.write(f"Test Case {i+1}:\n")
                f.write(f"  Ground Truth: {labels[i]} ({'Normal' if labels[i] == 0 else 'Arrhythmia'})\n")
                f.write(f"  Prediction: {predictions[i]} ({'Normal' if predictions[i] == 0 else 'Arrhythmia'})\n")
                f.write(f"  Match: {'✓' if labels[i] == predictions[i] else '✗'}\n")
                f.write(f"  Input (first 10 samples): {segments[i][:10]}\n")
                f.write("\n")
        
        print("✓ Saved text format")
    
    def save_hex_format(self, segments, labels, predictions, save_dir):
        """Save inputs in hexadecimal format for hardware"""
        with open(os.path.join(save_dir, 'inputs.hex'), 'w') as f:
            for segment in segments:
                # Convert each sample to fixed-point hex
                for sample in segment:
                    fixed_val = self.float_to_fixed(sample)
                    # Write as 16-bit hex (4 hex digits)
                    hex_val = fixed_val & 0xFFFF  # Ensure 16-bit
                    f.write(f"{hex_val:04X}\n")
        
        # Save expected outputs
        with open(os.path.join(save_dir, 'outputs.hex'), 'w') as f:
            for pred in predictions:
                f.write(f"{pred}\n")
        
        print("✓ Saved hex format")
    
    def save_verilog_format(self, segments, labels, predictions, save_dir):
        """Save as Verilog testbench stimulus"""
        with open(os.path.join(save_dir, 'testbench_stimulus.v'), 'w') as f:
            f.write("// Auto-generated testbench stimulus\n")
            f.write("// ECG Arrhythmia Detection Test Vectors\n\n")
            
            f.write("module test_vectors;\n")
            f.write("  // Test case parameters\n")
            f.write(f"  parameter NUM_CASES = {len(segments)};\n")
            f.write("  parameter INPUT_SIZE = 180;\n\n")
            
            # Write test inputs as memory
            f.write("  // Test inputs (fixed-point Q8.8)\n")
            f.write("  reg [15:0] test_inputs [0:NUM_CASES*INPUT_SIZE-1];\n")
            f.write("  reg [0:0] expected_outputs [0:NUM_CASES-1];\n\n")
            
            f.write("  initial begin\n")
            
            # Write input data
            idx = 0
            for i, segment in enumerate(segments):
                for sample in segment:
                    fixed_val = self.float_to_fixed(sample)
                    hex_val = fixed_val & 0xFFFF
                    f.write(f"    test_inputs[{idx}] = 16'h{hex_val:04X};\n")
                    idx += 1
            
            f.write("\n")
            
            # Write expected outputs
            for i, pred in enumerate(predictions):
                f.write(f"    expected_outputs[{i}] = 1'b{pred};\n")
            
            f.write("  end\n")
            f.write("endmodule\n")
        
        print("✓ Saved Verilog format")

def main():
    print("="*60)
    print("TEST VECTOR GENERATION")
    print("="*60)
    
    generator = TestVectorGenerator()
    
    # Generate 50 test vectors
    segments, labels, predictions = generator.generate_vectors(num_cases=50)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total test cases: {len(segments)}")
    print(f"Normal cases: {np.sum(labels == 0)}")
    print(f"Arrhythmia cases: {np.sum(labels == 1)}")
    print(f"Correct predictions: {np.sum(labels == predictions)}")
    print(f"Accuracy: {np.mean(labels == predictions)*100:.2f}%")
    print("="*60)
    
    print("\n✅ Test vector generation complete!")
    print("\nFiles created:")
    print("  - test_vectors.npz (NumPy format)")
    print("  - test_vectors.txt (Human-readable)")
    print("  - inputs.hex (Hardware input)")
    print("  - outputs.hex (Expected outputs)")
    print("  - testbench_stimulus.v (Verilog testbench)")

if __name__ == "__main__":
    main()