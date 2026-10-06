"""Evaluate a trained U-Net on a held-out binary obstacle-segmentation set.

The validation directory must follow the same layout as the training data:
    data_val/images/frame_XXXXXX.png
    data_val/masks/frame_XXXXXX.png

Example:
    env\Scripts\python.exe test_unet.py --data_dir data_val

Outputs are written to ``unet_evaluation/`` by default.  This script never
changes the model weights or any training data.
"""

import argparse
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config
from unet import UNet


class ValidationDataset(Dataset):
    """Loads only image/mask pairs with identical filenames."""

    def __init__(self, data_dir, size=config.IMG_SIZE):
        images_dir = os.path.join(data_dir, "images")
        masks_dir = os.path.join(data_dir, "masks")
        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            raise FileNotFoundError(
                f"Expected '{images_dir}' and '{masks_dir}'. "
                "Pass --data_dir data_val for the held-out data."
            )

        image_names = {name for name in os.listdir(images_dir)
                       if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))}
        mask_names = {name for name in os.listdir(masks_dir)
                      if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))}
        self.names = sorted(image_names & mask_names)
        if not self.names:
            raise ValueError("No matching image/mask filenames were found.")
        self.images_dir, self.masks_dir, self.size = images_dir, masks_dir, size

        missing_masks = image_names - mask_names
        missing_images = mask_names - image_names
        if missing_masks or missing_images:
            print(f"[warning] ignoring {len(missing_masks)} images without masks and "
                  f"{len(missing_images)} masks without images")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(os.path.join(self.images_dir, name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.masks_dir, name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Could not read validation pair '{name}'.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0)[None, ...]
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask), name


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def average_precision_from_hist(pos_hist, neg_hist):
    """Compute pixel-wise AP from score histograms without retaining all pixels."""
    true_positive = np.cumsum(pos_hist[::-1])
    false_positive = np.cumsum(neg_hist[::-1])
    positives = pos_hist.sum()
    negatives = neg_hist.sum()
    recall = true_positive / max(positives, 1)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    fpr = false_positive / max(negatives, 1)

    # Interpolated AP: precision is made non-increasing from high to low recall.
    recall_full = np.r_[0.0, recall]
    precision_full = np.r_[1.0, precision]
    precision_envelope = np.maximum.accumulate(precision_full[::-1])[::-1]
    ap = float(np.sum((recall_full[1:] - recall_full[:-1]) * precision_envelope[1:]))
    return recall_full, precision_full, np.r_[0.0, fpr], ap


def save_plots(output_dir, recall, precision, fpr, ious, latencies, samples, metrics):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="tab:blue", label=f"AP = {metrics['pixel_ap']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pixel-wise Precision-Recall Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=160)
    plt.close()

    tpr = recall
    auc = float(np.trapz(tpr, fpr))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="tab:orange", label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Pixel-wise ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(ious, bins=20, color="tab:green", edgecolor="white")
    axes[0].axvline(np.mean(ious), color="black", linestyle="--", label=f"mean = {np.mean(ious):.3f}")
    axes[0].set(title="Per-image Foreground IoU", xlabel="IoU", ylabel="Images", xlim=(0, 1))
    axes[0].legend()
    axes[1].hist(latencies, bins=min(20, max(5, len(latencies))), color="tab:purple", edgecolor="white")
    axes[1].axvline(np.mean(latencies), color="black", linestyle="--",
                    label=f"mean = {np.mean(latencies):.2f} ms")
    axes[1].set(title="Inference Latency", xlabel="Milliseconds per image", ylabel="Images")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "iou_and_latency_distributions.png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    confusion = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    image = axes[0].imshow(confusion, cmap="Blues")
    axes[0].set_xticks([0, 1], ["Predicted free", "Predicted obstacle"])
    axes[0].set_yticks([0, 1], ["Actual free", "Actual obstacle"])
    axes[0].set_title("Pixel Confusion Matrix")
    for row in range(2):
        for col in range(2):
            axes[0].text(col, row, f"{confusion[row, col]:,}", ha="center", va="center")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    names = ["Precision", "Recall", "F1", "IoU", "Dice", "Accuracy"]
    values = [metrics[key] for key in ["precision", "recall", "f1", "foreground_iou", "dice", "accuracy"]]
    axes[1].bar(names, values, color="tab:blue")
    axes[1].set(title="Validation Metrics", ylim=(0, 1))
    axes[1].tick_params(axis="x", rotation=35)
    for idx, value in enumerate(values):
        axes[1].text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_summary.png"), dpi=160)
    plt.close(fig)

    if samples:
        fig, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)))
        axes = np.atleast_2d(axes)
        for row, (name, rgb, target, probability, prediction) in enumerate(samples):
            overlay = rgb.copy()
            overlay[prediction] = 0.45 * overlay[prediction] + 0.55 * np.array([255, 0, 0])
            for axis, image_data, title, cmap in [
                (axes[row, 0], rgb, name, None),
                (axes[row, 1], target, "Ground truth", "gray"),
                (axes[row, 2], probability, "Obstacle probability", "magma"),
                (axes[row, 3], overlay, "Prediction overlay", None),
            ]:
                axis.imshow(image_data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
                axis.set_title(title)
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "sample_predictions.png"), dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_val", help="held-out image/mask dataset root")
    parser.add_argument("--weights", default=config.UNET_WEIGHTS_PATH, help="trained U-Net .pt file")
    parser.add_argument("--output_dir", default="unet_evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability threshold for binary metrics")
    parser.add_argument("--samples", type=int, default=6, help="number of visual prediction examples to save")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"No U-Net weights at '{args.weights}'.")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = ValidationDataset(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    criterion = nn.BCELoss(reduction="sum")

    tp = fp = fn = tn = 0
    total_loss = total_pixels = 0
    latency_ms, foreground_ious, samples = [], [], []
    bins = 1000
    pos_hist, neg_hist = np.zeros(bins, dtype=np.int64), np.zeros(bins, dtype=np.int64)

    with torch.no_grad():
        for index in range(len(dataset)):
            image, target, name = dataset[index]
            image_batch = image.unsqueeze(0).to(device)
            target_batch = target.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            probability = model(image_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms.append((time.perf_counter() - started) * 1000.0)

            total_loss += float(criterion(probability, target_batch).item())
            total_pixels += target.numel()
            prob = probability.squeeze().cpu().numpy()
            truth = target.squeeze().numpy() >= 0.5
            predicted = prob >= args.threshold
            tp += int(np.logical_and(predicted, truth).sum())
            fp += int(np.logical_and(predicted, ~truth).sum())
            fn += int(np.logical_and(~predicted, truth).sum())
            tn += int(np.logical_and(~predicted, ~truth).sum())

            foreground_ious.append(safe_divide(
                np.logical_and(predicted, truth).sum(), np.logical_or(predicted, truth).sum()
            ))
            score_bins = np.minimum((prob.ravel() * bins).astype(np.int32), bins - 1)
            pos_hist += np.bincount(score_bins[truth.ravel()], minlength=bins)
            neg_hist += np.bincount(score_bins[~truth.ravel()], minlength=bins)

            if len(samples) < args.samples:
                samples.append((
                    name,
                    np.transpose(image.numpy(), (1, 2, 0)),
                    truth,
                    prob,
                    predicted,
                ))

    recall_curve, precision_curve, fpr_curve, pixel_ap = average_precision_from_hist(pos_hist, neg_hist)
    foreground_iou = safe_divide(tp, tp + fp + fn)
    background_iou = safe_divide(tn, tn + fp + fn)
    metrics = {
        "validation_images": len(dataset),
        "threshold": args.threshold,
        "bce_loss": safe_divide(total_loss, total_pixels),
        "accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "foreground_iou": foreground_iou,
        "background_iou": background_iou,
        "mean_iou": (foreground_iou + background_iou) / 2.0,
        "pixel_ap": pixel_ap,
        # With one foreground class (obstacle), mAP is the same value as AP.
        "mAP": pixel_ap,
        "mean_per_image_foreground_iou": float(np.mean(foreground_ious)),
        "latency_mean_ms": float(np.mean(latency_ms)),
        "latency_median_ms": float(np.median(latency_ms)),
        "latency_p95_ms": float(np.percentile(latency_ms, 95)),
        "throughput_images_per_second": safe_divide(1000.0, np.mean(latency_ms)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)

    save_plots(output_dir=args.output_dir, recall=recall_curve, precision=precision_curve,
               fpr=fpr_curve, ious=foreground_ious, latencies=latency_ms,
               samples=samples, metrics=metrics)

    print(f"Validated {len(dataset)} images on {device.type.upper()}")
    print(f"BCE loss: {metrics['bce_loss']:.6f}")
    print(f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F1/Dice: {metrics['f1']:.4f}")
    print(f"Foreground IoU: {metrics['foreground_iou']:.4f}  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Pixel AP / single-class mAP: {metrics['pixel_ap']:.4f}")
    print(f"Latency: mean={metrics['latency_mean_ms']:.2f} ms  p95={metrics['latency_p95_ms']:.2f} ms  "
          f"throughput={metrics['throughput_images_per_second']:.2f} images/s")
    print(f"Saved metrics and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
