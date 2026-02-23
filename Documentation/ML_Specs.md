# ECG Arrhythmia Detector - ML Specifications

## Model Architecture

**Input:** 180 samples (1D ECG segment)
**Output:** Binary classification (0=Normal, 1=Arrhythmia)

### Layer-by-Layer Breakdown:
```
Layer 1: Conv1D
- Input channels: 1
- Output channels: 16
- Kernel size: 5
- Stride: 2
- Padding: 2
- Output size: 90 samples

Layer 2: Conv1D
- Input channels: 16
- Output channels: 32
- Kernel size: 5
- Stride: 2
- Padding: 2
- Output size: 45 samples

Layer 3: Conv1D
- Input channels: 32
- Output channels: 64
- Kernel size: 5
- Stride: 2
- Padding: 2
- Output size: 23 samples

Layer 4: Global Average Pooling
- Input: (64, 23)
- Output: (64, 1)

Layer 5: Fully Connected
- Input: 64
- Output: 2
```

## Fixed-Point Format

**Data Format:** Q8.8 (8 integer bits, 8 fractional bits)
- Range: -128.0 to +127.996
- Resolution: 1/256 = 0.00390625

**Conversion:**
- Float to Fixed: `fixed = int(float_value * 256)`
- Fixed to Float: `float_value = fixed / 256`

## Performance Targets

- **Accuracy:** 96.4% (quantized INT8)
- **Latency Target:** <10ms per inference
- **Throughput Target:** >100 inferences/second
- **Power Target:** <3W

## Test Vectors

- **Total cases:** 50
- **Normal cases:** 25
- **Arrhythmia cases:** 25
- **Expected accuracy:** 98%

## Memory Requirements

- **Weights:** ~13K parameters
- **Activations:** Calculate per layer
- **Input buffer:** 180 samples

## Contact

For questions about ML model, contact: [Your Name]
```

---

## 💬 **MESSAGE TO SEND:**

Copy this and send:
```
Hey! ML pipeline complete. Here's everything you need:

🎯 CRITICAL FILES (GitHub repo):
- data/test_vectors/inputs.hex (RTL test inputs)
- data/test_vectors/outputs.hex (expected results)
- data/test_vectors/testbench_stimulus.v (ready-to-use testbench)

📊 MODEL SPECS:
Architecture: 3 Conv layers (stride=2) + GAP + 1 FC
- Conv1: 1→16 channels, kernel=5, stride=2, output=90
- Conv2: 16→32 channels, kernel=5, stride=2, output=45
- Conv3: 32→64 channels, kernel=5, stride=2, output=23
- GAP: 64×23 → 64×1
- FC: 64→2 (classification)

Total parameters: 13,346

🔢 DATA FORMAT:
- Fixed-point: Q8.8 (8 int bits, 8 frac bits)
- Input: 180 samples per inference
- Output: 1 bit (0=Normal, 1=Arrhythmia)

✅ PERFORMANCE:
- Accuracy: 96.4% (quantized INT8)
- Test vector accuracy: 98% (50 cases)
- Target latency: <10ms
- Target throughput: >100 inf/sec

📁 ALL FILES:
Clone repo: https://github.com/Viraj281105/ecg_fpga_accelerator
Everything's in Code/ directory

Questions? Ping me! 🚀