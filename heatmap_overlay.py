import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def overlay_heatmap_with_manual_roi(original_img_path, snr_map_path, alpha=0.3, save_path=None):
    """
    Opens the original image and heatmap, allows the user to manually select a region
    (using the mouse), and overlays the heatmap only in that selected region.

    Parameters:
        original_img_path (str): Path to the hand image (e.g., roi_ref_visual.png)
        snr_map_path (str): Path to the heatmap (e.g., SNR_MAP_FRQ_0.png)
        alpha (float): Transparency value for overlay blending (0 = only hand, 1 = only heatmap)
        save_path (str): Full path to save the final overlay image (optional)
    """

    # Load hand image and convert to RGB
    hand_img_bgr = cv2.imread(original_img_path)
    if hand_img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {original_img_path}")
    hand_img = cv2.cvtColor(hand_img_bgr, cv2.COLOR_BGR2RGB)

    # Load heatmap (may be RGBA or RGB)
    snr_img = plt.imread(snr_map_path)
    if snr_img is None:
        raise FileNotFoundError(f"Could not read heatmap: {snr_map_path}")
    if snr_img.shape[2] == 4:
        snr_img = snr_img[:, :, :3]

    # Resize heatmap to match hand image if needed
    if snr_img.shape[:2] != hand_img.shape[:2]:
        snr_img = cv2.resize(snr_img, (hand_img.shape[1], hand_img.shape[0]))

    # User selects rectangular region
    cv2.namedWindow("Select Region to Overlay", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select Region to Overlay", hand_img_bgr)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)
    mask = np.zeros(hand_img.shape[:2], dtype=bool)
    mask[y:y+h, x:x+w] = True

    # Normalize heatmap if needed
    if snr_img.max() <= 1.0:
        snr_img = (snr_img * 255).astype(np.uint8)

    # Blend only selected region
    blended = hand_img.copy()
    for c in range(3):
        blended[:, :, c][mask] = (
            alpha * snr_img[:, :, c][mask] + (1 - alpha) * hand_img[:, :, c][mask]
        ).astype(np.uint8)

    # Show and optionally save result
    plt.figure(figsize=(12, 6))
    plt.imshow(blended)
    plt.axis('off')
    plt.title("Overlay on Selected ROI")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"✅ Overlay saved at: {save_path}")
    plt.show()
