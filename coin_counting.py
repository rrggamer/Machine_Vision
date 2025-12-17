import cv2
import numpy as np

# ----------------------------
# Config (tune here)
# ----------------------------
TARGET_SIZE = (500, 500)  # (w, h)

YELLOW_HSV_LOWER = np.array([20, 100, 100])
YELLOW_HSV_UPPER = np.array([35, 255, 255])

# NOTE: your blue threshold is applied on a BGR image (normalized),
# so these are BGR bounds (not HSV).
BLUE_BGR_LOWER = np.array([180, 145, 0])
BLUE_BGR_UPPER = np.array([255, 255, 150])

# Morphology kernels
K_Y_OPEN  = np.ones((10, 10), np.uint8)
K_Y_CLOSE = np.ones((2, 2), np.uint8)
K_Y_ERODE = np.ones((3, 3), np.uint8)

K_B_ERODE1 = np.ones((22, 22), np.uint8)
K_B_DILATE = np.ones((5, 1), np.uint8)
K_B_OPEN   = np.ones((5, 5), np.uint8)
K_B_ERODE2 = np.flipud(np.eye(7, dtype=np.uint8))

# Filter tiny blobs so you count real coins only
MIN_AREA_Y = 300
MIN_AREA_B = 300

SHOW_DEBUG = True


# ----------------------------
# Helpers
# ----------------------------
def normalize_illumination_bgr(img_bgr: np.ndarray, scale: float = 255.0, blur_ksize=(51, 51)) -> np.ndarray:
    """Flat-field illumination normalization: (img / blurred_gray_bg) * scale."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, blur_ksize, 0).astype(np.float32)
    bg = np.clip(bg, 1, 255)

    bg3 = cv2.merge([bg, bg, bg])  # float32 3-ch
    img_f = img_bgr.astype(np.float32)

    norm = (img_f / bg3) * scale
    return np.clip(norm, 0, 255).astype(np.uint8)


def filter_contours_by_area(contours, min_area: float):
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def show_win(name, img, w=800, h=600):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, img)


# ----------------------------
# Processing
# ----------------------------
def yellow_mask_from_image(im_bgr: np.ndarray) -> np.ndarray:
    """Yellow segmentation using HSV on normalized image."""
    im_norm = normalize_illumination_bgr(im_bgr, scale=255.0)
    hsv = cv2.cvtColor(im_norm, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  K_Y_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_Y_CLOSE)
    mask = cv2.erode(mask, K_Y_ERODE)
    return mask, im_norm


def blue_mask_from_image(im_bgr: np.ndarray) -> np.ndarray:
    """Blue segmentation using BGR threshold on normalized image (your original logic)."""
    im_norm_blue = normalize_illumination_bgr(im_bgr, scale=190.0)

    er1 = cv2.erode(im_norm_blue, K_B_ERODE1)
    dil = cv2.dilate(er1, K_B_DILATE)

    mask = cv2.inRange(dil, BLUE_BGR_LOWER, BLUE_BGR_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_B_OPEN)
    mask = cv2.erode(mask, K_B_ERODE2)
    return mask, im_norm_blue


def coin_counting(filename: str):
    im = cv2.imread(filename)
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {filename}")

    im = cv2.resize(im, TARGET_SIZE)

    # --- Yellow ---
    mask_y, norm_y = yellow_mask_from_image(im)
    contours_y, _ = cv2.findContours(mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counted_y = filter_contours_by_area(contours_y, MIN_AREA_Y)

    # --- Blue ---
    mask_b, norm_b = blue_mask_from_image(im)
    contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counted_b = filter_contours_by_area(contours_b, MIN_AREA_B)

    # --- Visualization: draw ONLY what we count ---
    vis = im.copy()
    cv2.drawContours(vis, counted_y, -1, (255, 0, 0), 3)
    cv2.drawContours(vis, counted_b, -1, (255, 0, 255), 3)

    yellow = len(counted_y)
    blue = len(counted_b)

    if SHOW_DEBUG:
        show_win("Original + Counted Contours", vis)
        show_win("Mask Yellow", mask_y)
        show_win("Mask Blue", mask_b)
        show_win("Norm (Yellow scale=255)", norm_y)
        show_win("Norm (Blue scale=190)", norm_b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return yellow, blue


# ----------------------------
# Main loop
# ----------------------------
for i in range(1, 10):
    path = rf"Dataset\CoinCounting\coin{i}.jpg"
    y, b = coin_counting(path)
    print(i, ":", [y, b])
