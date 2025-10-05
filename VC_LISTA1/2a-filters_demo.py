#!/usr/bin/env python3
"""2a_filters_demo.py

Aplica h1 = Box 3x3 e h2 = Laplaciano 3x3 nas imagens de 'grupo 1' nas duas ordens:
 (i) h1*(h2*I)
 (ii) h2*(h1*I)

Também constrói o kernel combinado h1*h2 (convolução discreta de kernels) e aplica
 (h1*h2)*I. Salva imagens de saída, diferenças e histogramas em VC_LISTA1/results_2a/.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolve_kernels(k1, k2):
    # 2D convolution of small kernels (no flips since we want standard convolution)
    s1 = k1.shape
    s2 = k2.shape
    out_shape = (s1[0] + s2[0] - 1, s1[1] + s2[1] - 1)
    out = np.zeros(out_shape, dtype=np.float64)
    # perform convolution
    for i in range(s1[0]):
        for j in range(s1[1]):
            out[i:i+s2[0], j:j+s2[1]] += k1[i,j] * k2
    return out


def apply_and_save(img, kernel, filename):
    # filter2D with same depth as src
    filtered = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    cv2.imwrite(filename, filtered)
    return filtered


def plot_and_save_hist(orig, imgs, labels, out_base):
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.imshow(orig, cmap='gray')
    plt.title('Original (gray)')
    plt.axis('off')

    plt.subplot(2,1,2)
    for im, lab in zip(imgs, labels):
        plt.hist(im.ravel(), bins=256, range=(0,255), alpha=0.6, label=lab)
    plt.legend()
    plt.title('Histogramas')
    plt.tight_layout()
    plt.savefig(out_base + '_hist.png', dpi=150)
    plt.close()


def process_image(path, out_dir):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Não foi possível abrir', path)
        return
    base = os.path.splitext(os.path.basename(path))[0]

    # definindo kernels
    h1 = np.ones((3,3), dtype=np.float32) / 9.0  # Box 3x3
    h2 = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)  # Laplacian 4-neighbors

    # (i) h1*(h2*I)
    h2_I = cv2.filter2D(img, ddepth=-1, kernel=h2)
    h1_h2I = cv2.filter2D(h2_I, ddepth=-1, kernel=h1)

    # (ii) h2*(h1*I)
    h1_I = cv2.filter2D(img, ddepth=-1, kernel=h1)
    h2_h1I = cv2.filter2D(h1_I, ddepth=-1, kernel=h2)

    # kernel combinado (convolução discreta) h1*h2
    combined_kernel = convolve_kernels(h1, h2)
    # Note: filter2D expects kernel center; combined may be 5x5
    combined = cv2.filter2D(img, ddepth=-1, kernel=combined_kernel)

    # salvar imagens
    out_i = os.path.join(out_dir, base + '_h1_h2I.png')
    out_ii = os.path.join(out_dir, base + '_h2_h1I.png')
    out_comb = os.path.join(out_dir, base + '_h1convh2_I.png')
    cv2.imwrite(out_i, h1_h2I)
    cv2.imwrite(out_ii, h2_h1I)
    cv2.imwrite(out_comb, combined)

    # salvar diferença absoluta
    diff = cv2.absdiff(h1_h2I.astype(np.int16), h2_h1I.astype(np.int16)).astype(np.uint8)
    out_diff = os.path.join(out_dir, base + '_absdiff.png')
    cv2.imwrite(out_diff, diff)

    # salvar kernel visualização (normalized for display)
    kvis = combined_kernel.copy()
    # normalize to 0..255 for saving
    kmin, kmax = kvis.min(), kvis.max()
    if kmax - kmin != 0:
        kvis_n = ((kvis - kmin) / (kmax - kmin) * 255).astype(np.uint8)
    else:
        kvis_n = (kvis*0).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, base + '_combined_kernel.png'), kvis_n)

    # histograms
    plot_and_save_hist(img, [h1_h2I, h2_h1I, combined, diff], ['h1(h2(I))', 'h2(h1(I))', 'combined(I)', 'absdiff'], os.path.join(out_dir, base))

    print('Salvo para', base)


def main(folder='grupo 1'):
    base_dir = os.path.dirname(__file__)
    folder_path = os.path.join(base_dir, folder)
    out_dir = os.path.join(base_dir, 'results_2a')
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif'))]
    if not files:
        print('Nenhuma imagem encontrada em', folder_path)
        return

    # processar apenas a primeira imagem para atender "para uma das imagens"
    first = files[0]
    process_image(os.path.join(folder_path, first), out_dir)


if __name__ == '__main__':
    main('grupo 1')
