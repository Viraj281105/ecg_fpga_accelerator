import torch
import torch.nn as nn
import numpy as np
from ecg_cnn import ECG_1DCNN_Optimized  # ← FIXED
from torch.utils.data import DataLoader, TensorDataset
import os

class ModelQuantizer:
    """Quantize trained model to INT8 for FPGA deployment"""
    
    def __init__(self, model, device='cpu'):  # ← Use CPU for quantization
        self.model = model.to(device)
        self.device = device
        self.quantized_model = None
        
    def post_training_quantization(self):
        """Apply PyTorch's built-in post-training quantization"""
        print("Applying post-training quantization...")
        
        # Move to CPU for quantization
        self.model.cpu()
        self.model.eval()
        
        # Quantize
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv1d},  # Layers to quantize
            dtype=torch.qint8
        )
        
        print("✓ Model quantized to INT8")
        return self.quantized_model
    
    def evaluate_quantized(self, test_loader):
        """Evaluate quantized model accuracy"""
        self.quantized_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.quantized_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def compare_models(self, test_loader):
        """Compare FP32 vs INT8 accuracy"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # FP32 accuracy
        self.model.eval()
        self.model.cpu()
        correct_fp32 = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_fp32 += (predicted == labels).sum().item()
        
        fp32_acc = 100 * correct_fp32 / total
        
        # INT8 accuracy
        int8_acc = self.evaluate_quantized(test_loader)
        
        print(f"FP32 Accuracy: {fp32_acc:.2f}%")
        print(f"INT8 Accuracy: {int8_acc:.2f}%")
        print(f"Accuracy Drop: {fp32_acc - int8_acc:.2f}%")
        
        # Model size comparison
        fp32_size = self.get_model_size(self.model)
        int8_size = self.get_model_size(self.quantized_model)
        
        print(f"\nFP32 Model Size: {fp32_size:.2f} MB")
        print(f"INT8 Model Size: {int8_size:.2f} MB")
        print(f"Size Reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
        print("="*60)
        
        return fp32_acc, int8_acc
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        torch.save(model.state_dict(), 'temp_model.pth')
        size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
        os.remove('temp_model.pth')
        return size_mb
    
    def export_quantized_weights(self, save_dir='results/models'):
        """Export quantized weights for hardware implementation"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\nExporting quantized weights...")
        
        # Save quantized model
        torch.save(self.quantized_model.state_dict(), 
                   os.path.join(save_dir, 'quantized_model.pth'))
        
        # Extract and save individual layer weights
        weights_dict = {}
        for name, param in self.quantized_model.named_parameters():
            weights_dict[name] = param.detach().cpu().numpy()
        
        np.savez(os.path.join(save_dir, 'quantized_weights.npz'), **weights_dict)
        
        print(f"✓ Quantized weights saved to {save_dir}/")
        
        return weights_dict

def main():
    print("="*60)
    print("MODEL QUANTIZATION")
    print("="*60)
    
    # Load trained model
    print("\nLoading trained model...")
    model = ECG_1DCNN_Optimized()  # ← FIXED
    model.load_state_dict(torch.load('results/models/best_model.pth', 
                                     map_location='cpu'))  # ← Load to CPU
    print("✓ Model loaded")
    
    # Load test data
    print("\nLoading test data...")
    segments = np.load('data/processed/segments.npy')
    labels = np.load('data/processed/labels.npy')
    
    # Use last 20% as test set
    test_size = int(0.2 * len(segments))
    test_segments = torch.FloatTensor(segments[-test_size:])
    test_labels = torch.LongTensor(labels[-test_size:])
    
    test_dataset = TensorDataset(test_segments, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"✓ Test set: {len(test_segments)} samples")
    
    # Quantize
    quantizer = ModelQuantizer(model)
    quantized_model = quantizer.post_training_quantization()
    
    # Compare
    fp32_acc, int8_acc = quantizer.compare_models(test_loader)
    
    # Export
    weights = quantizer.export_quantized_weights()
    
    print("\n✅ Quantization complete!")
    print(f"Accuracy retained: {int8_acc/fp32_acc*100:.1f}%")

if __name__ == "__main__":
    main()