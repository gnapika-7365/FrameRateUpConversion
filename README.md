# FrameRateUpConversion

## Motion-Confidence-Guided Adaptive FIR Frame Rate Up-Conversion

### Project Overview
This project aims to develop a hardware-efficient Frame Rate Up-Conversion (FRUC) algorithm that generates intermediate video frames to improve video smoothness.

The proposed approach combines Block Matching and Sum of Absolute Differences (SAD) for motion estimation, motion-confidence estimation, and adaptive Finite Impulse Response (FIR) filtering for intermediate frame generation.

### Objectives
- Implement Block Matching and SAD-based motion estimation.
- Estimate motion confidence from matching errors.
- Develop motion-confidence-guided adaptive FIR interpolation.
- Evaluate image quality using PSNR and SSIM.
- Prepare the algorithm for FPGA implementation.

### Methodology
1. Input consecutive video frames.
2. Divide frames into blocks.
3. Estimate motion using Block Matching and SAD.
4. Calculate motion confidence.
5. Apply adaptive FIR interpolation.
6. Generate intermediate frames.
7. Evaluate interpolation quality.

### Technologies
- Python
- OpenCV
- NumPy
- Vimeo-90K Triplet Dataset
- Verilog RTL
- Xilinx Vivado

### Target Hardware
Zynq-7000 FPGA.

### Evaluation Metrics
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)
- FPGA resource utilization
- Processing latency

### Project Status
Python algorithm development and evaluation are in progress/planned. Verilog RTL development and FPGA implementation are future stages.

### Team Members
- G. Gnapika
- T. Dedeepya
