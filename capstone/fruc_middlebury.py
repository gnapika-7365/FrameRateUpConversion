
import cv2
import numpy as np

# Motion-Confidence-Guided Adaptive FIR Frame Interpolation
# Software prototype for later Zynq-7000 FPGA implementation

VIDEO = "data/input.mp4"
OUTPUT = "fruc_adaptive_fir.mp4"

BLOCK = 16
SEARCH = 8
MAX_WIDTH = 480

# Adaptive 3-tap FIR coefficient sets.
# Inputs: warped previous frame, motion-compensated midpoint,
#         warped next frame.
FIR_HIGH = (0.15, 0.70, 0.15)
FIR_MED  = (0.25, 0.50, 0.25)
FIR_LOW  = (0.50, 0.00, 0.50)


def sad_match(src, ref, block, search):
    """Return block motion vectors and match confidence."""
    h, w = src.shape
    flow = np.zeros((h, w, 2), np.float32)
    conf = np.zeros((h, w), np.float32)

    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = src[y:y+block, x:x+block].astype(np.int16)
            costs = []

            for dy in range(-search, search + 1, 2):
                for dx in range(-search, search + 1, 2):
                    yy, xx = y + dy, x + dx

                    if (yy < 0 or xx < 0 or
                        yy + block > h or xx + block > w):
                        continue

                    candidate = ref[yy:yy+block, xx:xx+block]
                    cost = np.abs(
                        patch - candidate.astype(np.int16)
                    ).mean()
                    costs.append((cost, dx, dy))

            costs.sort(key=lambda z: z[0])
            if not costs:
                continue

            best_cost, dx, dy = costs[0]
            second_cost = costs[1][0] if len(costs) > 1 else best_cost

            # High confidence when the best match clearly wins.
            confidence = (
                (second_cost - best_cost) /
                (second_cost + 1e-6)
            )
            confidence = float(np.clip(confidence, 0, 1))

            flow[y:y+block, x:x+block, 0] = dx
            flow[y:y+block, x:x+block, 1] = dy
            conf[y:y+block, x:x+block] = confidence

    # Smooth block boundaries to reduce abrupt vector changes.
    flow[:, :, 0] = cv2.GaussianBlur(
        flow[:, :, 0], (5, 5), 0
    )
    flow[:, :, 1] = cv2.GaussianBlur(
        flow[:, :, 1], (5, 5), 0
    )
    conf = cv2.GaussianBlur(conf, (5, 5), 0)

    return flow, conf


def warp(image, map_x, map_y):
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def adaptive_fir(frame_a, frame_b):
    h, w = frame_a.shape[:2]

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # Forward and backward block motion estimation.
    flow_ab, conf_ab = sad_match(
        gray_a, gray_b, BLOCK, SEARCH
    )
    flow_ba, conf_ba = sad_match(
        gray_b, gray_a, BLOCK, SEARCH
    )

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    # Approximate the half-time position by warping each endpoint.
    a_x = x - 0.5 * flow_ab[:, :, 0]
    a_y = y - 0.5 * flow_ab[:, :, 1]
    b_x = x - 0.5 * flow_ba[:, :, 0]
    b_y = y - 0.5 * flow_ba[:, :, 1]

    warped_a = warp(frame_a, a_x, a_y).astype(np.float32)
    warped_b = warp(frame_b, b_x, b_y).astype(np.float32)

    # Initial motion-compensated midpoint.
    midpoint = 0.5 * warped_a + 0.5 * warped_b

    # Combine forward/backward confidence.
    confidence = np.clip(
        0.5 * (conf_ab + conf_ba), 0, 1
    )

    # Select a three-tap FIR set for each pixel.
    high = confidence >= 0.65
    medium = (confidence >= 0.25) & (confidence < 0.65)
    low = confidence < 0.25

    output = np.empty_like(midpoint)

    for mask, coeff in (
        (high, FIR_HIGH),
        (medium, FIR_MED),
        (low, FIR_LOW)
    ):
        c0, c1, c2 = coeff
        filtered = (
            c0 * warped_a +
            c1 * midpoint +
            c2 * warped_b
        )
        output[mask] = filtered[mask]

    return np.clip(output, 0, 255).astype(np.uint8), confidence


def main():
    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise FileNotFoundError(
            "Video not found. Check data/input.mp4"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Limit resolution during initial algorithm development.
    if original_w > MAX_WIDTH:
        w = MAX_WIDTH
        h = int(original_h * w / original_w)
    else:
        w, h = original_w, original_h

    output_fps = fps * 2
    out = cv2.VideoWriter(
        OUTPUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (w, h)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    ret, previous = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise ValueError("Input video has no readable frames.")

    previous = cv2.resize(previous, (w, h))
    pairs = 0

    while True:
        ret, current = cap.read()
        if not ret:
            break

        current = cv2.resize(current, (w, h))

        middle, confidence = adaptive_fir(previous, current)

        out.write(previous)
        out.write(middle)

        pairs += 1
        if pairs % 10 == 0:
            print(
                "Processed pairs:", pairs,
                "| Mean confidence:",
                round(float(confidence.mean()), 3)
            )

        previous = current

    out.write(previous)
    cap.release()
    out.release()

    print("\nAdaptive FIR prototype completed.")
    print("Input FPS:", fps)
    print("Output FPS:", output_fps)
    print("Output resolution:", w, "x", h)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
