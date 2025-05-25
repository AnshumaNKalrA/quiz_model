import os
import re
import json

def parse_count_from_filename(filename):
    """
    Parses the pixel count from a filename like '..._count_1234.png'.
    """
    match = re.search(r'_count_(\d+)', filename)
    if match:
        return int(match.group(1))
    print(f"Warning: Could not parse count from filename: {filename}. Expected format like '..._count_DIGITS.png'.")
    return None

def learn_precision_focused_okay_range(okay_counts, flagged_counts, min_coverage=0.9):
    """
    Learns a pixel count range that includes as many "okay" images as possible
    while minimizing flagged samples being included — favoring high precision.
    """
    best_min = None
    best_max = None
    best_precision = -1
    best_recall = 0

    total_okay = len(okay_counts)
    all_counts = sorted(set(okay_counts + flagged_counts))

    for i in range(len(all_counts)):
        for j in range(i, len(all_counts)):
            min_thresh = all_counts[i]
            max_thresh = all_counts[j]

            tp = sum(min_thresh <= c <= max_thresh for c in okay_counts)
            fp = sum(min_thresh <= c <= max_thresh for c in flagged_counts)

            if tp == 0:
                continue

            precision = tp / (tp + fp)
            recall = tp / total_okay

            if recall >= min_coverage and precision > best_precision:
                best_precision = precision
                best_recall = recall
                best_min = min_thresh
                best_max = max_thresh

    return best_min, best_max, best_precision, best_recall

def learn_optimal_pixel_range_from_filenames(okay_dir, flagged_dir):
    """
    Learns a high-precision pixel count range [min_thresh, max_thresh] for "okay" images.
    The model favors minimizing false positives (flagged images being inside the range).
    """
    okay_counts = []
    flagged_counts = []

    if not os.path.isdir(okay_dir):
        print(f"Error: 'Okay' directory not found: {okay_dir}")
        return None, None, 0.0
    print(f"Processing 'okay' images from: {okay_dir}")
    for filename in os.listdir(okay_dir):
        count = parse_count_from_filename(filename)
        if count is not None:
            okay_counts.append(count)

    if not os.path.isdir(flagged_dir):
        print(f"Error: 'Flagged' directory not found: {flagged_dir}")
        return None, None, 0.0
    print(f"Processing 'flagged' images from: {flagged_dir}")
    for filename in os.listdir(flagged_dir):
        count = parse_count_from_filename(filename)
        if count is not None:
            flagged_counts.append(count)

    if not okay_counts:
        print("Warning: No pixel counts from 'okay' files. Cannot define a meaningful 'okay' range.")
        return None, None, 0.0

    print(f"Learning high-precision range from {len(okay_counts)} 'okay' and {len(flagged_counts)} 'flagged' counts...")
    min_thresh, max_thresh, precision, recall = learn_precision_focused_okay_range(okay_counts, flagged_counts)

    if min_thresh is not None and max_thresh is not None:
        print("\n--- Learning Results ---")
        print(f"Best Okay Range Found: [{min_thresh}, {max_thresh}]")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f} (Fraction of 'okay' images covered)")
        return min_thresh, max_thresh, precision
    else:
        print("\nCould not determine a high-confidence range. Try adjusting the min_coverage or review your data.")
        return None, None, 0.0

# --- Configuration ---
OKAY_OPTIONS_DIR = "./option_box_samples/okay/"
FLAGGED_OPTIONS_DIR = "./option_box_samples/flagged/"
OUTPUT_CONFIG_FILE = "learned_option_thresholds.json"

def main():
    print("Starting learning process for option box pixel count range...")

    os.makedirs(OKAY_OPTIONS_DIR, exist_ok=True)
    os.makedirs(FLAGGED_OPTIONS_DIR, exist_ok=True)

    min_thresh, max_thresh, precision = learn_optimal_pixel_range_from_filenames(
        OKAY_OPTIONS_DIR, FLAGGED_OPTIONS_DIR
    )

    if min_thresh is not None and max_thresh is not None:
        print(f"\nSuccessfully learned high-precision range with precision: {precision:.4f}")
        print(f"Recommended Min Threshold for Option BG Pixels: {min_thresh}")
        print(f"Recommended Max Threshold for Option BG Pixels: {max_thresh}")

        threshold_config = {
            "opt_min_bg": min_thresh,
            "opt_max_bg": max_thresh,
            "precision": precision
        }
        try:
            with open(OUTPUT_CONFIG_FILE, 'w') as f:
                json.dump(threshold_config, f, indent=4)
            print(f"Learned thresholds saved to: {OUTPUT_CONFIG_FILE}")
        except IOError as e:
            print(f"Error saving thresholds to {OUTPUT_CONFIG_FILE}: {e}")

        print("\nNext Steps:")
        print("1. Use this high-precision range to confidently accept only safe 'okay' images.")
        print(f"2. Images outside [{min_thresh}, {max_thresh}] can be marked as 'IDK'.")
    else:
        print("\nFailed to learn an optimal range. Please check your image directories and filename formats.")

if __name__ == "__main__":
    main()
