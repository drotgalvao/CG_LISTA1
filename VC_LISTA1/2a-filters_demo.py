import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv_kernels(k1, k2):
    # convolution of small kernels (k1 * k2)
    return cv2.filter2D(k2, ddepth=-1, kernel=k1)


def ensure_float_kernel(k):
    kf = np.array(k, dtype=np.float64)
    # normalize sum if it's a box-like kernel (for display only)
    return kf


def process_one(path, out_dir):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("cannot open", path)
        return None
    base = os.path.splitext(os.path.basename(path))[0]

    # define kernels
    h1 = np.ones((3, 3), dtype=np.float64) / 9.0
    h2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    # compute in the two orders
    h2I = cv2.filter2D(img.astype(np.float64), ddepth=-1, kernel=h2)
    h1_h2I = cv2.filter2D(h2I, ddepth=-1, kernel=h1)

    h1I = cv2.filter2D(img.astype(np.float64), ddepth=-1, kernel=h1)
    h2_h1I = cv2.filter2D(h1I, ddepth=-1, kernel=h2)

    # combined kernel via discrete convolution (h1 * h2)
    combined_kernel = conv_kernels(h1, h2)
    combinedI = cv2.filter2D(img.astype(np.float64), ddepth=-1, kernel=combined_kernel)

    # differences
    absdiff_2_3 = np.abs(h1_h2I - h2_h1I)

    # save images (scaled back to 0..255 properly)
    def save_gray(array, path):
        # clip and convert
        a = np.clip(array, 0, 255).astype(np.uint8)
        cv2.imwrite(path, a)

    save_gray(h1_h2I, os.path.join(out_dir, base + "_h1h2I.png"))
    save_gray(h2_h1I, os.path.join(out_dir, base + "_h2h1I.png"))
    save_gray(combinedI, os.path.join(out_dir, base + "_combinedI.png"))
    save_gray(absdiff_2_3, os.path.join(out_dir, base + "_absdiff23.png"))

    # save kernels as text and as visual enlarged images
    np.savetxt(os.path.join(out_dir, base + "_kernel_h1.txt"), h1, fmt="%.6f")
    np.savetxt(os.path.join(out_dir, base + "_kernel_h2.txt"), h2, fmt="%.6f")
    np.savetxt(
        os.path.join(out_dir, base + "_kernel_h1convh2.txt"),
        combined_kernel,
        fmt="%.6f",
    )

    def save_kernel_vis(k, path):
        # normalize to 0..255 for display
        kmin, kmax = k.min(), k.max()
        if kmax - kmin == 0:
            kv = np.zeros_like(k, dtype=np.uint8)
        else:
            kv = ((k - kmin) / (kmax - kmin) * 255).astype(np.uint8)
        # upscale for visibility
        up = cv2.resize(
            kv, (kv.shape[1] * 60, kv.shape[0] * 60), interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(path, up)

    save_kernel_vis(h1, os.path.join(out_dir, base + "_kernel_h1_viz.png"))
    save_kernel_vis(h2, os.path.join(out_dir, base + "_kernel_h2_viz.png"))
    save_kernel_vis(
        combined_kernel, os.path.join(out_dir, base + "_kernel_combined_viz.png")
    )

    # histograms
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap="gray")
    plt.title(base + " original")
    plt.axis("off")
    plt.subplot(2, 1, 2)
    for a, label in [
        (h1_h2I, "h1(h2(I))"),
        (h2_h1I, "h2(h1(I))"),
        (combinedI, "(h1*h2)I"),
        (absdiff_2_3, "absdiff"),
    ]:
        plt.hist(a.ravel(), bins=256, range=(0, 255), alpha=0.6, label=label)
    plt.legend()
    plt.title("histograms")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, base + "_hist_compare.png"), dpi=150)
    plt.close()

    # compute norms to quantify differences
    diff_norm = np.linalg.norm(h1_h2I - h2_h1I)
    combined_minus_ordered_norm = np.linalg.norm(combinedI - h1_h2I) + np.linalg.norm(
        combinedI - h2_h1I
    )

    return {
        "base": base,
        "diff_norm": float(diff_norm),
        "combined_minus_ordered_norm": float(combined_minus_ordered_norm),
    }


def main():
    base_dir = os.path.dirname(__file__)
    folder = os.path.join(base_dir, "grupo 1")
    out = os.path.join(base_dir, "results_2a")
    os.makedirs(out, exist_ok=True)

    files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".bmp", ".tif", ".jpeg"))
    ]
    if not files:
        print("no images in", folder)
        return

    report = []
    for f in files:
        path = os.path.join(folder, f)
        print("processing", path)
        r = process_one(path, out)
        if r:
            report.append(r)

    # save simple report
    with open(os.path.join(out, "report_2a.txt"), "w") as fh:
        fh.write("base\tdiff_norm\tcombined_minus_ordered_norm\n")
        for r in report:
            fh.write(
                f"{r['base']}\t{r['diff_norm']:.6f}\t{r['combined_minus_ordered_norm']:.6f}\n"
            )

    print("done. outputs in", out)


if __name__ == "__main__":
    main()
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
            out[i : i + s2[0], j : j + s2[1]] += k1[i, j] * k2
    return out


def apply_and_save(img, kernel, filename):
    # filter2D with same depth as src
    filtered = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    cv2.imwrite(filename, filtered)
    return filtered


def plot_and_save_hist(orig, imgs, labels, out_base):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(orig, cmap="gray")
    plt.title("Original (gray)")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    for im, lab in zip(imgs, labels):
        plt.hist(im.ravel(), bins=256, range=(0, 255), alpha=0.6, label=lab)
    plt.legend()
    plt.title("Histogramas")
    plt.tight_layout()
    plt.savefig(out_base + "_hist.png", dpi=150)
    plt.close()


def process_image(path, out_dir):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Não foi possível abrir", path)
        return
    base = os.path.splitext(os.path.basename(path))[0]

    # definindo kernels
    h1 = np.ones((3, 3), dtype=np.float32) / 9.0  # Box 3x3
    h2 = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
    )  # Laplacian 4-neighbors

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
    out_i = os.path.join(out_dir, base + "_h1_h2I.png")
    out_ii = os.path.join(out_dir, base + "_h2_h1I.png")
    out_comb = os.path.join(out_dir, base + "_h1convh2_I.png")
    cv2.imwrite(out_i, h1_h2I)
    cv2.imwrite(out_ii, h2_h1I)
    cv2.imwrite(out_comb, combined)

    # salvar diferença absoluta
    diff = cv2.absdiff(h1_h2I.astype(np.int16), h2_h1I.astype(np.int16)).astype(
        np.uint8
    )
    out_diff = os.path.join(out_dir, base + "_absdiff.png")
    cv2.imwrite(out_diff, diff)

    # ---- parte b) |I - lowpass(I)| usando Box 3x3 como passa-baixa ----
    lp_box = cv2.filter2D(img, ddepth=-1, kernel=h1)
    abs_diff_lp = cv2.absdiff(img, lp_box)
    out_lp_diff = os.path.join(out_dir, base + "_abs_I_minus_box.png")
    cv2.imwrite(out_lp_diff, abs_diff_lp)

    # salvar kernel high-pass = delta - box (visualizar)
    # delta impulse matching combined_kernel size
    kh, kw = combined_kernel.shape
    delta = np.zeros_like(combined_kernel)
    cy, cx = kh // 2, kw // 2
    delta[cy, cx] = 1.0
    # place 3x3 box kernel centered in padded kernel
    box_padded = np.zeros_like(combined_kernel)
    bh, bw = h1.shape
    start_y = cy - bh // 2
    start_x = cx - bw // 2
    box_padded[start_y : start_y + bh, start_x : start_x + bw] = h1
    highpass_kernel = delta - box_padded
    hp_min, hp_max = highpass_kernel.min(), highpass_kernel.max()
    if hp_max - hp_min != 0:
        hp_vis = ((highpass_kernel - hp_min) / (hp_max - hp_min) * 255).astype(np.uint8)
    else:
        hp_vis = (highpass_kernel * 0).astype(np.uint8)
    # salvar versão visual ampliada do kernel high-pass
    hp_vis_path = os.path.join(out_dir, base + "_highpass_kernel.png")
    # upscale para visualização (nearest to keep discrete blocks)
    up_scale = 60
    hp_vis_up = cv2.resize(
        hp_vis,
        (hp_vis.shape[1] * up_scale, hp_vis.shape[0] * up_scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(hp_vis_path.replace(".png", "_viz.png"), hp_vis_up)
    # salvar também valores numéricos do kernel
    np.savetxt(
        os.path.join(out_dir, base + "_highpass_kernel.txt"),
        highpass_kernel,
        fmt="%.6f",
        delimiter="\t",
    )

    # salvar kernel visualização (normalized for display)
    kvis = combined_kernel.copy()
    # normalize to 0..255 for saving
    kmin, kmax = kvis.min(), kvis.max()
    if kmax - kmin != 0:
        kvis_n = ((kvis - kmin) / (kmax - kmin) * 255).astype(np.uint8)
    else:
        kvis_n = (kvis * 0).astype(np.uint8)
    # salvar versão visual ampliada do kernel combinado
    kvis_path = os.path.join(out_dir, base + "_combined_kernel.png")
    up_scale = 60
    kvis_up = cv2.resize(
        kvis_n,
        (kvis_n.shape[1] * up_scale, kvis_n.shape[0] * up_scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(kvis_path.replace(".png", "_viz.png"), kvis_up)
    # salvar valores numéricos do kernel combinado
    np.savetxt(
        os.path.join(out_dir, base + "_combined_kernel.txt"),
        combined_kernel,
        fmt="%.6f",
        delimiter="\t",
    )

    # histograms
    plot_and_save_hist(
        img,
        [h1_h2I, h2_h1I, combined, diff],
        ["h1(h2(I))", "h2(h1(I))", "combined(I)", "absdiff"],
        os.path.join(out_dir, base),
    )

    print("Salvo para", base)


def main(folder="grupo 1"):
    base_dir = os.path.dirname(__file__)
    folder_path = os.path.join(base_dir, folder)
    out_dir = os.path.join(base_dir, "results_2a")
    os.makedirs(out_dir, exist_ok=True)

    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    ]
    if not files:
        print("Nenhuma imagem encontrada em", folder_path)
        return

    # processar apenas a primeira imagem para atender "para uma das imagens"
    first = files[0]
    process_image(os.path.join(folder_path, first), out_dir)


if __name__ == "__main__":
    main("grupo 1")
