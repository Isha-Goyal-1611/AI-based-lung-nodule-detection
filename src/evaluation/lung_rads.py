"""
src/evaluation/lung_rads.py

Maps a malignancy probability (0-1) from MalignancyClassifier to an
illustrative Lung-RADS-style category.

DISCLAIMER: these thresholds (0.3 / 0.6 / 0.8) are simplified stand-ins,
not clinically validated cutoffs. Real Lung-RADS categorization involves
nodule size, growth over time, and criteria beyond a single probability
score. This is an illustrative mapping for demonstration purposes only —
state this plainly wherever this output is shown to a user.
"""


def malignancy_to_lungrads(score: float) -> str:
    """
    score: malignancy probability in [0, 1] from MalignancyClassifier's output.
    Returns an illustrative Lung-RADS-style category string.
    """
    if score < 0.3:
        return "Lung-RADS 1-2 (Routine follow-up)"
    elif score < 0.6:
        return "Lung-RADS 3 (3-month follow-up CT)"
    elif score < 0.8:
        return "Lung-RADS 4A (PET scan recommended)"
    else:
        return "Lung-RADS 4B (Biopsy recommended)"


if __name__ == "__main__":
    # Quick sanity check across the threshold boundaries
    test_scores = [0.1, 0.29, 0.3, 0.45, 0.59, 0.6, 0.75, 0.79, 0.8, 0.95]
    for s in test_scores:
        print(f"  {s:.2f} -> {malignancy_to_lungrads(s)}")
