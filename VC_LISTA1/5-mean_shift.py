import os
import cv2
import numpy as np
import csv
from collections import Counter


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def load_images_from(folder):
    imgs = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            imgs.append((fn, os.path.join(folder, fn)))
    return imgs


def resize_for_test(img, max_dim=512):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_dim:
        return img
    scale = max_dim / s
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def jaccard(a, b):
    # a,b are binary (0/255) uint8
    A = a > 0
    B = b > 0
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / union if union > 0 else 0.0


def run_grid_on_image(
    fn, path, out_dir, sp_list, sr_list, canny_low, canny_high, test_scale=512
):
    img = cv2.imread(path)
    if img is None:
        print("Failed to load", path)
        return []

    img_small = resize_for_test(img, max_dim=test_scale)
    gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    orig_edges = cv2.Canny(gray_small, canny_low, canny_high)
    results = []

    base_name = os.path.splitext(fn)[0]
    img_out_folder = os.path.join(out_dir, base_name)
    ensure_dir(img_out_folder)

    # save original resized and its edges for reference
    cv2.imwrite(
        os.path.join(img_out_folder, f"{base_name}_orig_resized.png"), img_small
    )
    cv2.imwrite(os.path.join(img_out_folder, f"{base_name}_orig_edges.png"), orig_edges)

    for sp in sp_list:
        for sr in sr_list:
            filtered = cv2.pyrMeanShiftFiltering(img_small, sp, sr)
            filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(filtered_gray, canny_low, canny_high)

            edge_count = int((edges > 0).sum())
            orig_count = int((orig_edges > 0).sum())
            jac = jaccard(orig_edges, edges)

            tag = f"sp{sp}_sr{sr}"
            cv2.imwrite(
                os.path.join(img_out_folder, f"{base_name}_ms_{tag}.png"), filtered
            )
            cv2.imwrite(
                os.path.join(img_out_folder, f"{base_name}_ms_{tag}_edges.png"), edges
            )

            results.append(
                {
                    "image": fn,
                    "sp": sp,
                    "sr": sr,
                    "orig_edges": orig_count,
                    "ms_edges": edge_count,
                    "jaccard": float(jac),
                    "out_filtered": os.path.join(
                        img_out_folder, f"{base_name}_ms_{tag}.png"
                    ),
                    "out_edges": os.path.join(
                        img_out_folder, f"{base_name}_ms_{tag}_edges.png"
                    ),
                }
            )

    return results


def apply_config_to_full_image(
    path, out_path_filtered, out_path_edges, sp, sr, canny_low, canny_high
):
    img = cv2.imread(path)
    if img is None:
        print("Failed to load", path)
        return False
    filtered = cv2.pyrMeanShiftFiltering(img, sp, sr)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    cv2.imwrite(out_path_filtered, filtered)
    cv2.imwrite(out_path_edges, edges)
    return True


def main():
    base = os.path.dirname(__file__)
    group1 = os.path.join(base, "grupo 1")
    group2 = os.path.join(base, "grupo 2")

    out_root = os.path.join(base, "results_5")
    ensure_dir(out_root)
    out_group1 = os.path.join(out_root, "group1")
    out_group2 = os.path.join(out_root, "group2")
    ensure_dir(out_group1)
    ensure_dir(out_group2)

    # try to read best Canny from previous summary, fallback to 50,200
    canny_low, canny_high = 50, 200
    summary_file = os.path.join(base, "results_4", "summary.txt")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                txt = f.read()
            # look for pattern best_orig_majority=(..., ...)
            import re

            m = re.search(r"best_orig_majority=\(\s*(\d+)\s*,\s*(\d+)\s*\)", txt)
            if m:
                canny_low = int(m.group(1))
                canny_high = int(m.group(2))
        except Exception:
            pass

    print("Using Canny thresholds:", canny_low, canny_high)

    imgs1 = load_images_from(group1)
    if not imgs1:
        print("No images found in", group1)
        return

    sp_list = [5, 10, 20, 30]
    sr_list = [10, 20, 30, 40]

    csv_path = os.path.join(out_root, "mean_shift_grid_group1.csv")
    rows = []

    best_per_image = {}
    for fn, path in imgs1:
        print("Processing (grid) ", fn)
        res = run_grid_on_image(
            fn, path, out_group1, sp_list, sr_list, canny_low, canny_high
        )
        rows.extend(res)
        if res:
            # select best by max jaccard
            best = max(res, key=lambda r: r["jaccard"])
            best_per_image[fn] = (best["sp"], best["sr"], best["jaccard"])

    # save grid csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "sp",
                "sr",
                "orig_edges",
                "ms_edges",
                "jaccard",
                "out_filtered",
                "out_edges",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["image"],
                    r["sp"],
                    r["sr"],
                    r["orig_edges"],
                    r["ms_edges"],
                    r["jaccard"],
                    r["out_filtered"],
                    r["out_edges"],
                ]
            )

    # determine majority config
    counts = Counter((v[0], v[1]) for v in best_per_image.values())
    if counts:
        majority_config, majority_count = counts.most_common(1)[0]
        chosen_sp, chosen_sr = majority_config
    else:
        chosen_sp, chosen_sr = sp_list[0], sr_list[0]

    print("\nBest per image:")
    for k, v in best_per_image.items():
        print(k, "-> sp=%d sr=%d jaccard=%.3f" % (v[0], v[1], v[2]))

    print("\nChosen majority config for group2:", chosen_sp, chosen_sr)

    # apply chosen config to full images in group1 (save full-size result too) and group2
    applied_csv = os.path.join(out_root, "applied_chosen_config.csv")
    with open(applied_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "image", "sp", "sr", "out_filtered", "out_edges"])

        # group1 full images
        for fn, path in imgs1:
            base_name = os.path.splitext(fn)[0]
            outf = os.path.join(
                out_group1, f"{base_name}_ms_sp{chosen_sp}_sr{chosen_sr}.png"
            )
            oute = os.path.join(
                out_group1, f"{base_name}_ms_sp{chosen_sp}_sr{chosen_sr}_edges.png"
            )
            ok = apply_config_to_full_image(
                path, outf, oute, chosen_sp, chosen_sr, canny_low, canny_high
            )
            if ok:
                writer.writerow(["group1", fn, chosen_sp, chosen_sr, outf, oute])

        # group2
        if os.path.exists(group2):
            imgs2 = load_images_from(group2)
            for fn, path in imgs2:
                base_name = os.path.splitext(fn)[0]
                outf = os.path.join(
                    out_group2, f"{base_name}_ms_sp{chosen_sp}_sr{chosen_sr}.png"
                )
                oute = os.path.join(
                    out_group2, f"{base_name}_ms_sp{chosen_sp}_sr{chosen_sr}_edges.png"
                )
                ok = apply_config_to_full_image(
                    path, outf, oute, chosen_sp, chosen_sr, canny_low, canny_high
                )
                if ok:
                    writer.writerow(["group2", fn, chosen_sp, chosen_sr, outf, oute])
        else:
            print("No group2 folder found at", group2)

    print("\nDone. Results in", out_root)


if __name__ == "__main__":
    main()
