import os
import numpy as np
import cv2
from skimage import metrics
import matplotlib.pyplot as plt


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def ideal_lowpass_fft(img, cutoff):
    # img float 0..255
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros_like(img, dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (y - crow) ** 2 + (x - ccol) ** 2 <= cutoff * cutoff
    mask[mask_area] = 1
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def gaussian_lowpass_fft(img, sigma):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    gauss = np.exp(-((y - crow) ** 2 + (x - ccol) ** 2) / (2 * (sigma**2)))
    f = np.fft.fftshift(np.fft.fft2(img))
    f_f = f * gauss
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_f)))
    return np.clip(img_back, 0, 255).astype(np.uint8)


def evaluate_filters(folder, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = [
        f for f in os.listdir(folder) if f.lower().endswith((".bmp", ".png", ".jpg"))
    ]
    rows = []
    for f in files:
        path = os.path.join(folder, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # create noisy versions
        noisy_low = add_gaussian_noise(img, 5)
        noisy_high = add_gaussian_noise(img, 20)

        # spatial filters
        spatial_results = {}
        spatial_results["box3"] = cv2.blur(noisy_high, (3, 3))
        spatial_results["gauss5"] = cv2.GaussianBlur(noisy_high, (5, 5), 1.0)
        spatial_results["median5"] = cv2.medianBlur(noisy_high, 5)
        spatial_results["bilateral"] = cv2.bilateralFilter(noisy_high, 9, 75, 75)

        # frequency filters
        freq_results = {}
        freq_results["ideal_cut20"] = ideal_lowpass_fft(noisy_high, 20)
        freq_results["ideal_cut40"] = ideal_lowpass_fft(noisy_high, 40)
        freq_results["gauss_sigma10"] = gaussian_lowpass_fft(noisy_high, 10)

        # evaluate
        for name, res in {**spatial_results, **freq_results}.items():
            psnr = metrics.peak_signal_noise_ratio(img, res, data_range=255)
            ssim = metrics.structural_similarity(img, res, data_range=255)
            rows.append((f, name, psnr, ssim))

    # save CSV
    import csv

    with open(os.path.join(out_dir, "filter_compare.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "filter", "psnr", "ssim"])
        for r in rows:
            w.writerow(r)

    print("saved results to", out_dir)


def main():
    base = os.path.dirname(__file__)
    out = os.path.join(base, "results_3")
    # process group 1
    evaluate_filters(os.path.join(base, "grupo 1"), os.path.join(out, "group1"))
    # if group 2 exists
    g2 = os.path.join(base, "grupo 2")
    if os.path.isdir(g2):
        evaluate_filters(g2, os.path.join(out, "group2"))


if __name__ == "__main__":
    main()
