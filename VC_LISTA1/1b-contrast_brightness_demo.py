#!/usr/bin/env python3
"""contrast_brightness_demo.py

Aplica várias operações de brilho e contraste nas imagens da pasta "grupo 1"
e salva as imagens transformadas e os histogramas comparativos em
VC_LISTA1/results_cb/.

Operações aplicadas:
- Brilho: +50, -50, gamma correction (gamma=0.7 bright)
- Contraste: multiply 1.5, multiply 0.7, CLAHE

Este script não usa argparse: processa automaticamente as imagens em
`VC_LISTA1/grupo 1`.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_brightness_add(img, delta):
    return np.clip(img.astype(np.int16) + int(delta), 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    inv = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(img, table)


def apply_contrast_mul(img, alpha):
    return np.clip(img.astype(np.float32) * float(alpha), 0, 255).astype(np.uint8)


def apply_clahe(img, clip=2.0, tile=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)


def plot_and_save(img, transforms, base, out_dir):
    # img: original gray
    n = 1 + len(transforms)
    plt.figure(figsize=(4*n, 4))
    plt.subplot(1, n, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    for i, (name, timg) in enumerate(transforms, start=2):
        plt.subplot(1, n, i); plt.imshow(timg, cmap='gray'); plt.title(name); plt.axis('off')
    plt.tight_layout()
    img_path = os.path.join(out_dir, base + '_cb_images.png')
    plt.savefig(img_path, dpi=150)
    plt.close()

    # histograma comparativo
    plt.figure(figsize=(8,5))
    plt.subplot(2,1,1)
    plt.hist(img.ravel(), bins=256, range=(0,255), color='gray')
    plt.title(f'Histograma original — {base}')
    plt.subplot(2,1,2)
    for name, timg in transforms:
        plt.hist(timg.ravel(), bins=256, range=(0,255), alpha=0.6, label=name)
    plt.legend()
    plt.title('Histogramas das transformações')
    plt.tight_layout()
    hist_path = os.path.join(out_dir, base + '_cb_hist.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()


def process_folder(folder='grupo 1'):
    base_dir = os.path.dirname(__file__)
    folder_path = os.path.join(base_dir, folder)
    out_dir = os.path.join(base_dir, '1b-results')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(folder_path):
        print('Pasta não encontrada:', folder_path)
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif'))]
    if not files:
        print('Nenhuma imagem encontrada em', folder_path)
        return

    for f in files:
        path = os.path.join(folder_path, f)
        print('Processando', path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('Não foi possível abrir', path)
            continue

        base = os.path.splitext(f)[0]

        # preparar transformações
        transforms = []
        # brilho: +50, -50, gamma(0.7)
        transforms.append(('bright+50', apply_brightness_add(img, 50)))
        transforms.append(('bright-50', apply_brightness_add(img, -50)))
        transforms.append(('gamma0.7', apply_gamma(img, 0.7)))

        # contraste: *1.5, *0.7, CLAHE
        transforms.append(('contrastx1.5', apply_contrast_mul(img, 1.5)))
        transforms.append(('contrastx0.7', apply_contrast_mul(img, 0.7)))
        transforms.append(('clahe', apply_clahe(img, clip=2.0, tile=(8,8))))

        # salvar cada transformação individualmente também
        for name, timg in transforms:
            out_img_path = os.path.join(out_dir, f'{base}_{name}.png')
            cv2.imwrite(out_img_path, timg)

        # salvar montagens e histogramas
        plot_and_save(img, transforms, base, out_dir)

    print('Processamento concluído. Resultados em', out_dir)


if __name__ == '__main__':
    process_folder('grupo 1')
