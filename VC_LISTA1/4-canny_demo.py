import os
import cv2
import numpy as np
import csv


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run_grid_on_image(img, grid, out_prefix, out_dir):
    rows, cols = img.shape[:2]
    results = []
    for low, high in grid:
        edges = cv2.Canny(img, low, high)
        count = int(np.count_nonzero(edges))
        fn = f"{out_prefix}_canny_{low}_{high}.png"
        cv2.imwrite(os.path.join(out_dir, fn), edges)
        results.append({"low": low, "high": high, "count": count, "file": fn})
    return results


def choose_best_by_median(results):
    # choose combo whose edge count is closest to the median count (balanced)
    counts = [r["count"] for r in results]
    med = np.median(counts)
    best = min(results, key=lambda r: abs(r["count"] - med))
    return best, med


def process_folder(folder, out_base, grid, prefix, preprocess=False):
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    if not os.path.isdir(folder_path):
        return []
    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    ]
    report = []
    for f in files:
        path = os.path.join(folder_path, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        orig_img = img.copy()
        if preprocess:
            # Gaussian blur as single pre-processing for all images
            img = cv2.GaussianBlur(img, (5, 5), 1.0)

        out_dir = os.path.join(out_base, prefix, os.path.splitext(f)[0])
        ensure_dir(out_dir)
        res = run_grid_on_image(img, grid, os.path.splitext(f)[0], out_dir)
        best, med = choose_best_by_median(res)
        report.append(
            {
                "image": f,
                "preprocess": preprocess,
                "best_low": best["low"],
                "best_high": best["high"],
                "best_count": best["count"],
                "median_count": int(med),
            }
        )
    return report


def main():
    base_dir = os.path.dirname(__file__)
    out_base = os.path.join(base_dir, "results_4")
    ensure_dir(out_base)

    # parameter grid (low, high)
    lows = [50, 100, 150]
    highs = [100, 150, 200]
    grid = [(l, h) for l in lows for h in highs if h > l]

    # process group 1 without preprocess
    rep1 = process_folder("grupo 1", out_base, grid, "orig", preprocess=False)
    # process group 1 with Gaussian blur preprocess
    rep2 = process_folder("grupo 1", out_base, grid, "blur", preprocess=True)

    # save report CSV
    rpt_path = os.path.join(out_base, "canny_report_group1.csv")
    with open(rpt_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "image",
                "preprocess",
                "best_low",
                "best_high",
                "best_count",
                "median_count",
            ]
        )
        for r in rep1 + rep2:
            writer.writerow(
                [
                    r["image"],
                    r["preprocess"],
                    r["best_low"],
                    r["best_high"],
                    r["best_count"],
                    r["median_count"],
                ]
            )

    # choose global best parameters per preprocess mode by majority voting across images (most frequent best combo)
    def majority_choice(report_list):
        combos = [(r["best_low"], r["best_high"]) for r in report_list]
        if not combos:
            return None
        vals, counts = np.unique(combos, axis=0, return_counts=True)
        idx = np.argmax(counts)
        return tuple(vals[idx])

    best_orig = majority_choice(rep1)
    best_blur = majority_choice(rep2)

    # apply chosen parameters to group 2 if exists
    group2_path = os.path.join(os.path.dirname(__file__), "grupo 2")
    applied = []
    if os.path.isdir(group2_path):
        files2 = [
            f
            for f in os.listdir(group2_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        ]
        for f in files2:
            path = os.path.join(group2_path, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # apply best_orig and best_blur (if available)
            out_dir = os.path.join(out_base, "group2")
            ensure_dir(out_dir)
            if best_orig:
                edges = cv2.Canny(img, best_orig[0], best_orig[1])
                cv2.imwrite(
                    os.path.join(
                        out_dir,
                        os.path.splitext(f)[0]
                        + f"_canny_orig_{best_orig[0]}_{best_orig[1]}.png",
                    ),
                    edges,
                )
            if best_blur:
                imgb = cv2.GaussianBlur(img, (5, 5), 1.0)
                edgesb = cv2.Canny(imgb, best_blur[0], best_blur[1])
                cv2.imwrite(
                    os.path.join(
                        out_dir,
                        os.path.splitext(f)[0]
                        + f"_canny_blur_{best_blur[0]}_{best_blur[1]}.png",
                    ),
                    edgesb,
                )
            applied.append(f)

    # summary text
    with open(os.path.join(out_base, "summary.txt"), "w") as fh:
        fh.write("Canny parameter sweep summary\n")
        fh.write(f"best_orig_majority={best_orig}\n")
        fh.write(f"best_blur_majority={best_blur}\n")
        fh.write(
            "Processed images: group1 orig and blur. If group2 existed, applied chosen params.\n"
        )

    print("Done. Results in", out_base)


if __name__ == "__main__":
    main()
