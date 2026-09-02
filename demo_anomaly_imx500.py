"""
TinyGLASS student — interactive IMX500 anomaly-detection demo (single panel).

Live feed with the on-sensor anomaly heatmap blended on top. A per-frame
image score (max of the 16x16 patch map) drives a GOOD / ANOMALY verdict.
Everything runs on the IMX500 NPU — no CPU inference.

Model: checkpoints/rpk/network.rpk  (mms_rpi GLASS student, 96 KB discriminator)

Controls:
  • + / =   : raise anomaly threshold
  • - / _   : lower anomaly threshold
  • h       : cycle view (overlay -> heatmap only -> raw)
  • q / ESC : quit
"""

import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

MODEL_PATH   = os.path.join(os.path.dirname(__file__),
                            "checkpoints", "rpk", "network.rpk")
WINDOW_NAME  = "TinyGLASS IMX500 Anomaly Detection"
DISPLAY_SIZE = (1280, 960)          # W x H — camera main stream
THRESHOLD    = 0.10                 # good mms_rpi maxes ~0.09; defects >0.3
VIS_MAX      = 0.50                 # heatmap saturates to red at this score

imx500 = None
picam2 = None

latest_frame = None
latest_scores = None                # 16x16 float map in [0,1)
last_request = None

threshold = THRESHOLD
view_mode = 0                       # 0 overlay, 1 heatmap only, 2 raw


def anomaly_callback(request: CompletedRequest):
    global latest_frame, latest_scores, last_request

    latest_frame = request.make_array("main")   # RGB888 config -> BGR array for cv2
    last_request = request

    outputs = imx500.get_outputs(request.get_metadata())
    if outputs is None:
        return
    latest_scores = np.squeeze(outputs[0]).astype(np.float32)   # 16x16 in [0,1)

    if os.environ.get("TINYGLASS_DEBUG"):
        now = time.time()
        if now - anomaly_callback._last > 1.0:
            anomaly_callback._last = now
            print(f"[dbg] scores shape={latest_scores.shape} "
                  f"min={latest_scores.min():.3f} max={latest_scores.max():.3f} "
                  f"mean={latest_scores.mean():.3f}", flush=True)


anomaly_callback._last = 0.0


def render_heatmap(scores: np.ndarray) -> np.ndarray:
    """16x16 score map -> smooth full-size JET heatmap (BGR)."""
    vis = np.clip(scores / VIS_MAX, 0.0, 1.0)
    small = (vis * 255).astype(np.uint8)
    big = cv2.resize(small, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=9)
    return cv2.applyColorMap(big, cv2.COLORMAP_JET)


if __name__ == "__main__":
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), np.uint8))
    cv2.waitKey(1)

    imx500 = IMX500(MODEL_PATH)
    intr = imx500.network_intrinsics or NetworkIntrinsics()
    # Feed the full frame (squashed to 256x256) so the 16x16 map covers the
    # whole display; no center-crop.
    intr.preserve_aspect_ratio = False
    intr.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": DISPLAY_SIZE, "format": "RGB888"},   # -> BGR array (correct in cv2)
        controls={"FrameRate": intr.inference_rate},
        buffer_count=8,
    )
    picam2.start(config, show_preview=False)
    picam2.pre_callback = anomaly_callback

    print("TinyGLASS IMX500 anomaly demo running.")
    print("  + / - threshold  |  h = view  |  ESC/q = quit")

    try:
        while True:
            if latest_frame is None:
                time.sleep(0.005)
                continue

            frame = latest_frame.copy()
            scores = latest_scores

            if scores is not None:
                score = float(scores.max())
                heat = render_heatmap(scores)

                if view_mode == 0:
                    disp = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
                elif view_mode == 1:
                    disp = heat
                else:
                    disp = frame

                is_anom = score >= threshold
                label = "ANOMALY" if is_anom else "GOOD"
                color = (0, 0, 220) if is_anom else (0, 180, 0)
                cv2.putText(disp, f"{label}   score={score:.3f}  thr={threshold:.2f}",
                            (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                # thick border cue when anomalous
                if is_anom:
                    cv2.rectangle(disp, (2, 2),
                                  (DISPLAY_SIZE[0] - 3, DISPLAY_SIZE[1] - 3), color, 4)
            else:
                disp = frame
                cv2.putText(disp, "Loading model on IMX500...",
                            (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            cv2.putText(disp, "TinyGLASS | +/- thr | h=view | q=quit",
                        (12, DISPLAY_SIZE[1] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('+'), ord('=')):
                threshold = min(1.0, round(threshold + 0.02, 2))
            elif key in (ord('-'), ord('_')):
                threshold = max(0.0, round(threshold - 0.02, 2))
            elif key == ord('h'):
                view_mode = (view_mode + 1) % 3

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
