
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from fruc_middlebury import adaptive_fir

DATASET = r"C:\Users\thall\Downloads\test-00000-of-00001.parquet"
df = pd.read_parquet(DATASET)

def decode_image(item):
    data = np.frombuffer(item["bytes"], dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode dataset image")
    return image

blend_psnr, blend_ssim = [], []
fir_psnr, fir_ssim = [], []

for i in range(min(10, len(df))):
    row = df.iloc[i]
    frame1 = decode_image(row["im1"])
    gt = decode_image(row["im2"])
    frame3 = decode_image(row["im3"])

    if frame1.shape != gt.shape or frame3.shape != gt.shape:
        print("Skipping size mismatch:", i + 1)
        continue

    # Baseline 1: simple average of the two input frames
    blended = cv2.addWeighted(frame1, 0.5, frame3, 0.5, 0)

    # Proposed method: motion-confidence-guided adaptive FIR
    generated, confidence = adaptive_fir(frame1, frame3)

    # Calculate PSNR and SSIM for both methods
    bp = peak_signal_noise_ratio(gt, blended, data_range=255)
    bs = structural_similarity(
        gt, blended, channel_axis=2, data_range=255
    )
    fp = peak_signal_noise_ratio(gt, generated, data_range=255)
    fs = structural_similarity(
        gt, generated, channel_axis=2, data_range=255
    )

    blend_psnr.append(bp)
    blend_ssim.append(bs)
    fir_psnr.append(fp)
    fir_ssim.append(fs)

    print(f"\nSample {i + 1}")
    print(f"Blending:   PSNR={bp:.2f} dB, SSIM={bs:.4f}")
    print(f"Adaptive FIR: PSNR={fp:.2f} dB, SSIM={fs:.4f}")

    if i == 0:
        cv2.imwrite("baseline_blending.png", blended)
        cv2.imwrite("adaptive_fir.png", generated)
        cv2.imwrite("comparison_ground_truth.png", gt)

if blend_psnr:
    print("\n========== AVERAGE RESULTS ==========")
    print(f"Samples evaluated: {len(blend_psnr)}")
    print(
        f"Blending   - PSNR: {np.mean(blend_psnr):.2f} dB, "
        f"SSIM: {np.mean(blend_ssim):.4f}"
    )
    print(
        f"Adaptive FIR - PSNR: {np.mean(fir_psnr):.2f} dB, "
        f"SSIM: {np.mean(fir_ssim):.4f}"
    )
    print("\nPositive PSNR/SSIM differences mean Adaptive FIR performed better.")
    print(f"PSNR difference: {np.mean(fir_psnr) - np.mean(blend_psnr):+.2f} dB")
    print(f"SSIM difference: {np.mean(fir_ssim) - np.mean(blend_ssim):+.4f}")
else:
    print("No samples evaluated.")
