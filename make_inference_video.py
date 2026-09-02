"""
Render a TinyGLASS inference video over a whole test set.

Runs the same inference path as `demo_imx500.py image` (ResNet18 backbone,
layer2+layer3, 256x256 input), but sweeps every test image of every class,
computes image/pixel AUROC, and renders an annotated 1280x720 mp4:

    Input | Anomaly map | Overlay | Ground truth

Usage:
  uv run python make_inference_video.py mvtec \
      --data-root /datasets/pbonazzi/tinyglass_mvtec \
      --checkpoints checkpoints --out results/video/tinyglass_mvtec.mp4

  uv run python make_inference_video.py mms \
      --data-root /datasets/pbonazzi/tinyglass_mmdataset \
      --checkpoints checkpoints --out results/video/tinyglass_mms.mp4
"""

import argparse
import glob
import os
import subprocess
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from torchvision import transforms

import demo_imx500 as D
from glass import IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------------
# Canvas constants
# ---------------------------------------------------------------------------
W, H = 1280, 720
PANEL = 288
GAP = 20
LEFT = (W - (4 * PANEL + 3 * GAP)) // 2
TOP = 138
BG = (22, 22, 26)
FG = (232, 232, 236)
DIM = (140, 140, 148)
ACCENT = (210, 160, 60)      # BGR: amber
RED = (60, 60, 235)
GREEN = (90, 190, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_D = cv2.FONT_HERSHEY_DUPLEX

# datasets/mvtec.py resizes to --resize (256 in every run script) and center-crops
# to 256, except toothbrush/wood which are resized to round(256*329/288) first.
# demo_imx500.py instead resizes straight to 256x256.
EVAL_RESIZE = round(256 * 329 / 288)
WIDE_RESIZE_CLASSES = ("toothbrush", "wood")
GEOM = GEOM_M = TENSOR = None


def make_transforms(mode, cls=""):
    """(image geometry, mask geometry, model input) transforms, as in datasets/mvtec.py."""
    nearest = transforms.InterpolationMode.NEAREST
    if mode == "dataset":
        r = EVAL_RESIZE if cls in WIDE_RESIZE_CLASSES else 256
        geom = transforms.Compose([transforms.Resize(r), transforms.CenterCrop(256)])
        geom_m = transforms.Compose([transforms.Resize(r, interpolation=nearest),
                                     transforms.CenterCrop(256)])
    else:
        geom = transforms.Resize((256, 256))
        geom_m = transforms.Resize((256, 256), interpolation=nearest)
    tensor = transforms.Compose([geom, transforms.ToTensor(),
                                 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return geom, geom_m, tensor


def load_image(path):
    return cv2.cvtColor(np.array(GEOM(Image.open(path).convert("RGB"))), cv2.COLOR_RGB2BGR)


def load_mask(path):
    return np.array(GEOM_M(Image.open(path).convert("L")))


MVTEC_CLASSES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                 "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                 "transistor", "wood", "zipper"]


def txt(img, s, org, scale=0.5, color=FG, thick=1, font=FONT):
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def txt_center(img, s, cx, y, scale=0.5, color=FG, thick=1, font=FONT):
    (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
    txt(img, s, (int(cx - tw / 2), y), scale, color, thick, font)


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

def list_class_images(root, cls, dataset):
    """Return [(img_path, mask_path|None, defect_name)] for one class' test set."""
    if dataset == "mvtec":
        test_dir = os.path.join(root, "images", "test", cls)
        mask_dir = os.path.join(root, "masks", "test", cls)
    else:
        test_dir = os.path.join(root, cls, "test")
        mask_dir = os.path.join(root, cls, "ground_truth")

    items = []
    for defect in sorted(os.listdir(test_dir), key=lambda d: (d != "good", d)):
        for p in sorted(glob.glob(os.path.join(test_dir, defect, "*"))):
            m = None
            if defect != "good":
                stem = os.path.splitext(os.path.basename(p))[0]
                for cand in (f"{stem}_mask.png", f"{stem}.png"):
                    c = os.path.join(mask_dir, defect, cand)
                    if os.path.exists(c):
                        m = c
                        break
            items.append((p, m, defect))
    return items


def find_checkpoint(ckpt_root, cls):
    pat = os.path.join(ckpt_root, "models", "backbone_0", f"mvtec_{cls}", "ckpt_best_*.pth")
    paths = glob.glob(pat)
    if not paths:
        return None
    # highest epoch = the checkpoint reported in the README table
    return max(paths, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))


# ---------------------------------------------------------------------------
# Inference over one class
# ---------------------------------------------------------------------------

def run_class(cls, items, ckpt, device):
    glass = D.load_checkpoint(D.build_glass(device), ckpt, device)
    scores, segs, lat = [], [], []
    for img_path, _, defect in items:
        pil = Image.open(img_path).convert("RGB")
        t0 = time.perf_counter()
        s, seg = D.infer(glass, TENSOR(pil).unsqueeze(0), device)
        lat.append((time.perf_counter() - t0) * 1e3)
        scores.append(s)
        segs.append(seg.astype(np.float32))
    return np.array(scores), segs, np.array(lat)


def image_metrics(scores, labels):
    auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    # threshold at best F1 over the class' own test set
    best_f1, best_t = -1.0, float(np.median(scores))
    for t in np.unique(scores):
        pred = scores >= t
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return auroc, best_t


def pixel_auroc(segs, masks):
    ys, ps = [], []
    for seg, m in zip(segs, masks):
        if m is None:
            continue
        ys.append(m.ravel() > 0)
        ps.append(seg.ravel())
    if not ys:
        return float("nan")
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    if y.max() == y.min():
        return float("nan")
    return roc_auc_score(y, p)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def base_canvas(dataset_title, subtitle):
    f = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(f, (0, 0), (W, 76), (32, 32, 38), -1)
    txt(f, "TinyGLASS", (34, 46), 1.0, ACCENT, 2, FONT_D)
    txt(f, "Real-Time Self-Supervised In-Sensor Anomaly Detection", (215, 46), 0.52, DIM)
    (tw, _), _ = cv2.getTextSize(dataset_title, FONT, 0.6, 1)
    txt(f, dataset_title, (W - 34 - tw, 32), 0.6, FG)
    (tw2, _), _ = cv2.getTextSize(subtitle, FONT, 0.45, 1)
    txt(f, subtitle, (W - 34 - tw2, 56), 0.45, DIM)
    return f


def panel_xy(i):
    return LEFT + i * (PANEL + GAP), TOP


def paste(frame, i, img):
    x, y = panel_xy(i)
    frame[y:y + PANEL, x:x + PANEL] = cv2.resize(img, (PANEL, PANEL))
    cv2.rectangle(frame, (x - 1, y - 1), (x + PANEL, y + PANEL), (70, 70, 78), 1)


def colorbar(frame, x, y, w, h, vmax):
    grad = np.linspace(0, 255, w).astype(np.uint8)[None, :].repeat(h, 0)
    frame[y:y + h, x:x + w] = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    cv2.rectangle(frame, (x - 1, y - 1), (x + w, y + h), (70, 70, 78), 1)
    txt(frame, "0.00", (x, y + h + 14), 0.38, DIM)
    txt(frame, f"{vmax:.2f}", (x + w - 30, y + h + 14), 0.38, DIM)
    txt(frame, "anomaly score", (x + w // 2 - 42, y - 6), 0.38, DIM)


def score_timeline(f, scores, labels, cur, thr, smax):
    """Bar chart of every image-level score in the class, current image marked."""
    x0, y0 = LEFT, 540
    w, h = 4 * PANEL + 3 * GAP, 104
    cv2.rectangle(f, (x0, y0), (x0 + w, y0 + h), (30, 30, 36), -1)
    n = len(scores)
    bw = max(1, int(w / n))
    for i, (sc, lb) in enumerate(zip(scores, labels)):
        bx = x0 + int(i * w / n)
        bh = int(min(sc / smax, 1.0) * (h - 6))
        col = RED if lb else GREEN
        cv2.rectangle(f, (bx, y0 + h - bh), (bx + bw, y0 + h), col, -1)
    ty = y0 + h - int(min(thr / smax, 1.0) * (h - 6))
    cv2.line(f, (x0, ty), (x0 + w, ty), (235, 235, 235), 1)
    txt(f, "threshold", (x0 + w - 68, ty - 5), 0.36, (235, 235, 235))
    cx = x0 + int(cur * w / n)
    cv2.line(f, (cx, y0 - 4), (cx, y0 + h + 4), ACCENT, 1)
    cv2.drawMarker(f, (cx, y0 - 6), ACCENT, cv2.MARKER_TRIANGLE_DOWN, 9, 2)
    txt(f, "image-level score over the full test split", (x0, y0 - 10), 0.42, DIM)
    txt(f, "normal", (x0 + 320, y0 - 10), 0.42, GREEN)
    txt(f, "defective", (x0 + 392, y0 - 10), 0.42, RED)


def render_frame(dataset_title, subtitle, cls, item, score, seg, vmax, thr,
                 lat_ms, auroc, p_auroc, idx, total, gt_kind="mask",
                 all_scores=None, all_labels=None, cur=0):
    img_path, mask_path, defect = item
    f = base_canvas(dataset_title, subtitle)

    orig = load_image(img_path)

    seg_u8 = np.clip(seg / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(seg_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.55, heat, 0.45, 0)
    pred_mask = (seg >= thr).astype(np.uint8)
    cnts, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)

    # MMS ships only image-level labels (its ground_truth files are 10x10
    # placeholder blocks at a fixed corner), MVTec ships real pixel masks.
    if gt_kind == "mask":
        gt_cap = "ground truth"
        if mask_path is not None:
            gt = load_mask(mask_path)
            gt_vis = orig.copy()
            gt_vis[gt > 0] = (0.35 * gt_vis[gt > 0] + 0.65 * np.array([70, 70, 250])).astype(np.uint8)
            gtc, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_vis, gtc, -1, (90, 90, 255), 1)
        else:
            gt_vis = (orig * 0.35).astype(np.uint8)
            txt_center(gt_vis, "no defect", 128, 132, 0.6, (170, 170, 170), 1)
    else:
        gt_cap = "ground truth (image-level)"
        gt_vis = (orig * 0.35).astype(np.uint8)
        lab = "DEFECTIVE" if defect != "good" else "NORMAL"
        txt_center(gt_vis, lab, 128, 124, 0.7, RED if defect != "good" else GREEN, 2, FONT_D)
        txt_center(gt_vis, "no pixel labels in this release", 128, 152, 0.42, (170, 170, 170), 1)

    for i, (img, cap) in enumerate([(orig, "input"), (heat, "anomaly map"),
                                    (overlay, "overlay + prediction"),
                                    (gt_vis, gt_cap)]):
        paste(f, i, img)
        x, _ = panel_xy(i)
        txt(f, cap, (x + 2, TOP - 10), 0.46, DIM)

    # ---- class / file line ----
    txt(f, cls, (LEFT, 108), 0.72, FG, 2, FONT_D)
    (cw, _), _ = cv2.getTextSize(cls, FONT_D, 0.72, 2)
    txt(f, f"/ {defect}", (LEFT + cw + 14, 108), 0.55, ACCENT)
    name = os.path.basename(img_path)
    (nw, _), _ = cv2.getTextSize(name, FONT, 0.45, 1)
    txt(f, name, (LEFT + 4 * PANEL + 3 * GAP - nw, 108), 0.45, DIM)

    # ---- verdict + score bar ----
    y0 = TOP + PANEL + 66
    is_anom = score >= thr
    gt_anom = defect != "good"
    label = "ANOMALY" if is_anom else "GOOD"
    col = RED if is_anom else GREEN
    cv2.rectangle(f, (LEFT, y0 - 26), (LEFT + 178, y0 + 14), col, -1)
    txt_center(f, label, LEFT + 89, y0 + 2, 0.72, (255, 255, 255), 2, FONT_D)

    correct = is_anom == gt_anom
    txt(f, "ground truth:", (LEFT + 200, y0 - 6), 0.46, DIM)
    txt(f, "anomalous" if gt_anom else "normal", (LEFT + 312, y0 - 6), 0.46, FG)
    txt(f, "correct" if correct else "misclassified", (LEFT + 200, y0 + 16), 0.46,
        GREEN if correct else RED)

    bx, by, bw = LEFT + 660, y0 - 18, 290
    smax = max(vmax, thr * 1.6, 1e-6)
    cv2.rectangle(f, (bx, by), (bx + bw, by + 20), (48, 48, 54), -1)
    fill = int(min(score / smax, 1.0) * bw)
    cv2.rectangle(f, (bx, by), (bx + fill, by + 20), col, -1)
    tx = bx + int(min(thr / smax, 1.0) * bw)
    cv2.line(f, (tx, by - 6), (tx, by + 26), (255, 255, 255), 1)
    txt(f, f"thr {thr:.3f}", (tx - 26, by - 12), 0.38, DIM)
    txt(f, f"score {score:.3f}", (bx, by + 38), 0.5, FG)
    txt(f, f"{lat_ms:.0f} ms / frame  (CPU)", (bx + 160, by + 38), 0.46, DIM)

    colorbar(f, panel_xy(1)[0], TOP + PANEL + 12, PANEL, 10, vmax)

    if all_scores is not None:
        score_timeline(f, all_scores, all_labels, cur, thr, smax)

    # ---- footer: metrics + progress ----
    cv2.rectangle(f, (0, H - 58), (W, H), (32, 32, 38), -1)
    m = f"image AUROC {auroc * 100:.2f}%" if auroc == auroc else "image AUROC n/a"
    if p_auroc == p_auroc:
        m += f"     pixel AUROC {p_auroc * 100:.2f}%"
    txt(f, f"{cls}:  {m}", (34, H - 34), 0.52, FG)
    txt(f, f"{idx}/{total}", (W - 100, H - 34), 0.5, DIM)
    pw = int((idx / max(total, 1)) * W)
    cv2.rectangle(f, (0, H - 4), (pw, H), ACCENT, -1)
    return f


def title_card(lines, dataset_title, subtitle):
    f = base_canvas(dataset_title, subtitle)
    y = 300
    for i, (s, scale, color, thick) in enumerate(lines):
        txt_center(f, s, W // 2, y, scale, color, thick, FONT_D)
        y += int(46 * scale + 26)
    return f


def summary_card(rows, dataset_title, subtitle):
    f = base_canvas(dataset_title, subtitle)
    txt(f, "Results on the full test set", (LEFT, 140), 0.8, FG, 2, FONT_D)
    cols = [LEFT, LEFT + 260, LEFT + 470, LEFT + 700]
    txt(f, "class", (cols[0], 190), 0.5, DIM)
    txt(f, "images", (cols[1], 190), 0.5, DIM)
    txt(f, "image AUROC", (cols[2], 190), 0.5, DIM)
    txt(f, "pixel AUROC", (cols[3], 190), 0.5, DIM)
    y = 218
    step = 26 if len(rows) > 10 else 40
    for cls, n, a, p in rows:
        txt(f, cls, (cols[0], y), 0.5, FG)
        txt(f, str(n), (cols[1], y), 0.5, FG)
        txt(f, f"{a * 100:.2f}%" if a == a else "-", (cols[2], y), 0.5, FG)
        txt(f, f"{p * 100:.2f}%" if p == p else "-", (cols[3], y), 0.5, FG)
        y += step
    va = [a for _, _, a, _ in rows if a == a]
    vp = [p for _, _, _, p in rows if p == p]
    cv2.line(f, (cols[0], y - 18), (cols[3] + 140, y - 18), (70, 70, 78), 1)
    txt(f, "mean", (cols[0], y + 8), 0.55, ACCENT, 2)
    if va:
        txt(f, f"{np.mean(va) * 100:.2f}%", (cols[2], y + 8), 0.55, ACCENT, 2)
    if vp:
        txt(f, f"{np.mean(vp) * 100:.2f}%", (cols[3], y + 8), 0.55, ACCENT, 2)
    return f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def subsample(items, max_n):
    """Evenly sample up to max_n items, keeping every defect type represented."""
    if max_n <= 0 or len(items) <= max_n:
        return list(range(len(items)))
    by_defect = {}
    for i, (_, _, d) in enumerate(items):
        by_defect.setdefault(d, []).append(i)
    keep = []
    for d, idxs in by_defect.items():
        n = max(2, round(max_n * len(idxs) / len(items)))
        n = min(n, len(idxs))
        keep += [idxs[j] for j in np.linspace(0, len(idxs) - 1, n).round().astype(int)]
    return sorted(set(keep))


def to_h264(path):
    """OpenCV can only write mp4v here; re-encode so ordinary players accept it."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print("imageio-ffmpeg not installed - leaving the mp4v stream as is")
        return
    tmp = path + ".tmp.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path,
           "-c:v", "libx264", "-preset", "slow", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    if subprocess.call(cmd) == 0:
        os.replace(tmp, path)
    else:
        print("ffmpeg re-encode failed - keeping the mp4v stream")
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["mvtec", "mms"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=int, default=3, help="frames held per image")
    ap.add_argument("--max-per-class", type=int, default=40,
                    help="images shown per class (0 = all); metrics always use all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preprocess", choices=["dataset", "demo"], default="dataset",
                    help="'dataset' = the datasets/mvtec.py eval pipeline behind the "
                         "reported numbers, 'demo' = plain resize to 256 (demo_imx500.py)")
    args = ap.parse_args()

    device = torch.device(args.device)
    classes = args.classes or (MVTEC_CLASSES if args.dataset == "mvtec" else ["mms_rpi"])
    dataset_title = "MVTec AD" if args.dataset == "mvtec" else "MMS (M&Ms candies)"
    gt_kind = "mask" if args.dataset == "mvtec" else "marker"
    subtitle = "ResNet18 - layer2+layer3 - 256x256 - per-class checkpoint"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit(f"cannot open VideoWriter for {args.out}")

    def push(frame, n):
        for _ in range(n):
            vw.write(frame)

    push(title_card([("TinyGLASS inference", 1.1, FG, 2),
                     (dataset_title, 0.8, ACCENT, 2),
                     (f"{len(classes)} class(es) - anomaly localisation on the test split", 0.5, DIM, 1)],
                    dataset_title, subtitle), args.fps * 3)

    rows = []
    for ci, cls in enumerate(classes):
        ckpt = find_checkpoint(args.checkpoints, cls)
        if ckpt is None:
            print(f"[skip] no checkpoint for {cls}")
            continue
        global GEOM, GEOM_M, TENSOR
        GEOM, GEOM_M, TENSOR = make_transforms(args.preprocess, cls)
        items = list_class_images(args.data_root, cls, args.dataset)
        labels = np.array([0 if d == "good" else 1 for _, _, d in items])
        print(f"[{ci + 1}/{len(classes)}] {cls}: {len(items)} test images  ({os.path.basename(ckpt)})")

        scores, segs, lat = run_class(cls, items, ckpt, device)
        auroc, thr = image_metrics(scores, labels)
        masks = [None if m is None else load_mask(m) for (_, m, _) in items]
        p_auroc = pixel_auroc(segs, masks) if args.dataset == "mvtec" else float("nan")
        vmax = float(np.percentile(np.concatenate([s.ravel() for s in segs]), 99.9))
        vmax = max(vmax, thr * 1.2, 1e-3)
        rows.append((cls, len(items), auroc, p_auroc))
        print(f"      image AUROC {auroc * 100:.2f}  pixel AUROC {p_auroc * 100:.2f}  "
              f"thr {thr:.3f}  median {np.median(lat):.0f} ms")

        push(title_card([(cls, 1.3, FG, 2),
                         (f"{len(items)} test images - {os.path.basename(ckpt)}", 0.55, DIM, 1),
                         (f"image AUROC {auroc * 100:.2f}%" +
                          (f"   pixel AUROC {p_auroc * 100:.2f}%" if p_auroc == p_auroc else ""),
                          0.7, ACCENT, 2)],
                        dataset_title, subtitle), int(args.fps * 1.4))

        keep = subsample(items, args.max_per_class)
        for k, i in enumerate(keep):
            frame = render_frame(dataset_title, subtitle, cls, items[i], float(scores[i]),
                                 segs[i], vmax, thr, float(lat[i]), auroc, p_auroc,
                                 k + 1, len(keep), gt_kind, scores, labels, i)
            push(frame, args.hold)

    push(summary_card(rows, dataset_title, subtitle), args.fps * 5)
    vw.release()
    to_h264(args.out)
    dur = 0
    cap = cv2.VideoCapture(args.out)
    if cap.isOpened():
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    cap.release()
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, {dur:.0f}s)")


if __name__ == "__main__":
    main()
