import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def process_image(path, out_dir=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Não foi possível abrir", path)
        return

    base = os.path.splitext(os.path.basename(path))[0]

    # Global threshold manual
    T = 128
    _, thr_manual = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

    # Otsu
    _, thr_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive (mean) e adaptive (gaussian)
    thr_adapt_mean = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5
    )
    thr_adapt_gauss = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
    )

    results = [thr_manual, thr_otsu, thr_adapt_mean, thr_adapt_gauss]
    titles = [f"Global T={T}", "Otsu", "Adapt mean", "Adapt gauss"]

    # plot images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    for i, (res, t) in enumerate(zip(results, titles), start=2):
        plt.subplot(1, 5, i)
        plt.imshow(res, cmap="gray")
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()

    # histogramas
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.hist(img.ravel(), bins=256, range=(0, 255), color="gray")
    plt.title(f"Histograma — {base} (original)")
    plt.subplot(2, 1, 2)
    for res, t in zip(results, titles):
        plt.hist(res.ravel(), bins=256, range=(0, 255), alpha=0.5, label=t)
    plt.legend()
    plt.title("Histogramas das binarizações")
    plt.tight_layout()

    if out_dir:
        img_path = os.path.join(out_dir, base + "_results.png")
        hist_path = os.path.join(out_dir, base + "_hist.png")
        plt.figure(1)
        plt.savefig(img_path, dpi=150)
        plt.figure(2)
        plt.savefig(hist_path, dpi=150)
        plt.close("all")
        print("Salvo", img_path, hist_path)
    else:
        plt.show()


def main_process_folder(folder="grupo 1", out_dir=None):
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    if not os.path.isdir(folder_path):
        print("Pasta não encontrada:", folder_path)
        return
    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    ]
    if not files:
        print("Nenhuma imagem na pasta", folder_path)
        return
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for f in files:
        p = os.path.join(folder_path, f)
        process_image(p, out_dir)


if __name__ == "__main__":
    # processa automaticamente as imagens em VC_LISTA1/grupo 1
    out = os.path.join(os.path.dirname(__file__), "1a-results")
    main_process_folder("grupo 1", out_dir=out)
