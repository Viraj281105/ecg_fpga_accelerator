# Hardware Team Handoff

## Quick Start

1. Clone repo
2. Navigate to `Code/data/test_vectors/`
3. Use `inputs.hex` for RTL testbench
4. Compare RTL output against `outputs.hex`

## Test Vector Format

### inputs.hex
- Each line = 1 sample (16-bit hex)
- 180 lines = 1 test case
- 50 test cases total = 9,000 lines

Example:
```
00B4  ← Sample 1 of test case 1
00C2  ← Sample 2 of test case 1
...
```

### outputs.hex
- Each line = expected output
- 0 = Normal
- 1 = Arrhythmia

## Verification Process

1. Feed inputs.hex to your RTL
2. Capture RTL outputs
3. Compare against outputs.hex
4. Target: 98% match (49/50 correct)

## RTL Module Interface (Suggested)
```verilog
module ecg_accelerator (
    input clk,
    input rst,
    input [15:0] data_in,      // Q8.8 fixed-point
    input data_valid,
    output reg [0:0] result,   // 0=Normal, 1=Arrhythmia
    output reg result_valid
);
```

## Questions?

ML side: [Your contact]
Hardware side: [Partner contact]