
import cv2
import numpy as np
import pandas as pd
from fruc_middlebury import adaptive_fir

DATASET = r"C:\Users\thall\Downloads\test-00000-of-00001.parquet"
NUM_SAMPLES = 10

def decode_image(item):
    data = np.frombuffer(item["bytes"], dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode dataset image")
    return image

def calculate_metrics(gt, generated):
    # Normalize pixel values to [0, 1]
    x = gt.astype(np.float64) / 255.0
    y = generated.astype(np.float64) / 255.0

    error = x - y
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    psnr = float("inf") if mse == 0 else 20 * np.log10(1.0 / rmse)

    # SSIM calculated using an 11x11 Gaussian window
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    var_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    var_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    cov_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (
        (mu_x ** 2 + mu_y ** 2 + c1) *
        (var_x + var_y + c2)
    )

    ssim = float(np.mean(numerator / np.maximum(denominator, 1e-12)))
    return mse, mae, rmse, psnr, ssim

def main():
    df = pd.read_parquet(DATASET)
    count = min(NUM_SAMPLES, len(df))

    metrics = {
        "MSE": [], "MAE": [], "RMSE": [],
        "PSNR": [], "SSIM": [], "Confidence": []
    }

    for i in range(count):
        row = df.iloc[i]

        frame1 = decode_image(row["im1"])
        ground_truth = decode_image(row["im2"])
        frame3 = decode_image(row["im3"])

        if not (frame1.shape == ground_truth.shape == frame3.shape):
            print(f"Skipping sample {i + 1}: image sizes differ")
            continue

        # Generate intermediate frame using adaptive FIR
        generated, confidence = adaptive_fir(frame1, frame3)

        mse, mae, rmse, psnr, ssim = calculate_metrics(
            ground_truth, generated
        )

        metrics["MSE"].append(mse)
        metrics["MAE"].append(mae)
        metrics["RMSE"].append(rmse)
        metrics["PSNR"].append(psnr)
        metrics["SSIM"].append(ssim)
        metrics["Confidence"].append(float(np.mean(confidence)))

        print(f"\nSample {i + 1}")
        print(f"Generated normalized range: "
              f"{generated.min()/255:.4f} to {generated.max()/255:.4f}")
        print(f"Ground truth normalized range: "
              f"{ground_truth.min()/255:.4f} to {ground_truth.max()/255:.4f}")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM (Gaussian-window implementation): {ssim:.4f}")
        print(f"Mean confidence: {np.mean(confidence):.4f}")

        if not metrics["MSE"][:-1]:
            cv2.imwrite("input_first.png", frame1)
            cv2.imwrite("ground_truth.png", ground_truth)
            cv2.imwrite("input_last.png", frame3)
            cv2.imwrite("generated_middle.png", generated)

    if not metrics["MSE"]:
        print("No samples evaluated.")
        return

    print("\n========== AVERAGE RESULTS ==========")
    print("Samples evaluated:", len(metrics["MSE"]))
    for name, values in metrics.items():
        print(f"Mean {name}: {np.mean(values):.6f}")

if __name__ == "__main__":
    main()
