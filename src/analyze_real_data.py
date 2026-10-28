import pandas as pd
from pathlib import Path
import io

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"

REAL_DATASETS = {
    "ul_dd_zenodo": {
        "name": "UL-DD Multimodal Dataset",
        "url": "https://zenodo.org/record/9999999/files/UL-DD-extracted-features.csv",
        "desc": "Multimodal: video, biometric, behavioral. Natural imbalance",
        "source": "Zenodo (multimodal - video + EEG + posture)"
    },
    "nthu_ddd_kaggle": {
        "name": "NTHU-DDD Benchmark",
        "url": "https://www.kaggle.com/datasets/banudeep/nthuddd2",
        "desc": "36 subjects, frame-level labels, 65k+ frames",
        "source": "Kaggle (video dataset)"
    }
}


def analyze_structure_before_download():
    """Analyze expected data structure dari literatur."""
    print("=== EXPECTED REAL DATA STRUCTURE ===\n")
    
    print("UTA-RLDD Processed Features (GitHub):")
    print("  Columns: perclos, norm_avg_ear, head_pose_x/y/z, ...")
    print("  Rows: ~1.7M (60 subjects)")
    print("  Distribution: Alert 60-70%, Drowsy 15-25%, Distracted 10-20%")
    print("  Class imbalance: NATURAL (tidak dipaksakan)\n")
    
    print("UL-DD (Zenodo):")
    print("  Modalities: Video + EEG + Posture landmarks")
    print("  Participants: 19 (A-S)")
    print("  Sessions: Awake (A) vs Drowsy (D)")
    print("  Format: CSV + MP4")
    print("  Distribution: NATURALLY IMBALANCED\n")
    
    print("NTHU-DDD (Kaggle):")
    print("  Type: Video dataset (frame-level)")
    print("  Subjects: 36 (diverse: glasses, sunglasses, night, etc)")
    print("  Total frames: 65,929+")
    print("  Classes: Alert vs Drowsy (binary in raw form)")
    print("  Distribution: REALISTIC (more alert, less drowsy)\n")
    
    print("=== MAPPING STRATEGY ===\n")
    print("Real data → Pipeline columns:")
    print("  perclos (real)           → perclos (ours) ✓")
    print("  norm_avg_ear (real)      → ear_norm (ours) ✓")
    print("  head_pose_x/y (real)     → pitch_mean, yaw_mean (ours) ✓")
    print("  Label (alert/drowsy)     → label (ours) ✓")
    print()
    print("Priority: DROWSY recall > precision > alert/distracted")
    print("→ Use class_weight = {alert: 1, drowsy: 5, distracted: 1}")


def create_converter_template():
    """Template converter untuk berbagai format dataset."""
    template = '''
def convert_uta_rldd_csv(df):
    """Convert UTA-RLDD Processed CSV ke format kami."""
    converted = pd.DataFrame({
        "timestamp": range(len(df)),
        "subject": df.get("subject_id", "uta"),
        "ear_mean": df["norm_avg_ear"],
        "ear_std": 0.01,  # estimated
        "ear_norm": df["norm_avg_ear"],  # already normalized
        "mar_mean": df.get("norm_avg_mouth", 0.08),  # fallback
        "perclos": df["perclos"],
        "blink_rate": df.get("blink_count", 0),
        "max_closed_sec": df.get("max_eye_closure", 0),
        "pitch_mean": df["head_pose_pitch"],
        "pitch_std": 0.5,  # estimated
        "yaw_mean": df["head_pose_yaw"],
        "yaw_std": 1.0,  # estimated
        "label": df["state"].map({
            0: "alert", 1: "drowsy", 2: "low_vigilance"
        })
    })
    return converted

def convert_ul_dd_csv(df):
    """Convert UL-DD CSV ke format kami."""
    converted = pd.DataFrame({
        "timestamp": range(len(df)),
        "subject": df["participant_id"],
        "ear_mean": df["EAR"],
        "ear_std": 0.01,
        "ear_norm": df["EAR"] / 0.3,
        "mar_mean": df["MAR"],
        "perclos": df.get("PERCLOS", 0),
        "blink_rate": df.get("blink_freq", 0),
        "max_closed_sec": df.get("eye_closure_duration", 0),
        "pitch_mean": df.get("head_pitch", 0),
        "pitch_std": 0.5,
        "yaw_mean": df.get("head_yaw", 0),
        "yaw_std": 1.0,
        "label": df["session"].map({
            "A": "alert", "D": "drowsy"
        })
    })
    return converted
'''
    return template


def analyze_synthetic_vs_real():
    """Compare synthetic (sekarang) vs real data yang diharapkan."""
    print("\n=== SYNTHETIC (CURRENT) vs REAL (PROPOSED) ===\n")
    
    data = {
        "Aspect": [
            "Distribution",
            "Alert rows",
            "Drowsy rows",
            "Distracted rows",
            "Subjects",
            "Variability",
            "Realism",
            "Overfitting risk",
            "Production ready"
        ],
        "Synthetic (2748 rows)": [
            "33.3% each (FORCED)",
            "916",
            "916",
            "916",
            "3",
            "Low (generated)",
            "Low",
            "HIGH ⚠",
            "NO"
        ],
        "Real UTA-RLDD (1.7M rows)": [
            "60-70%, 15-25%, 10-20% (NATURAL)",
            "~1.02M",
            "~0.255M",
            "~0.17M",
            "60",
            "High (real world)",
            "High ✓",
            "LOW ✓",
            "YES ✓"
        ]
    }
    
    df_compare = pd.DataFrame(data)
    print(df_compare.to_string(index=False))
    print()
    print("Kesimpulan:")
    print("✓ Gunakan real data untuk model akurat")
    print("✓ Natural imbalance mencerminkan realitas (jarang tidur saat berkendara)")
    print("✓ Drowsy recall > precision untuk safety-critical")


if __name__ == "__main__":
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DATA STRATEGY: SYNTHETIC → REAL")
    print("=" * 70)
    print()
    
    analyze_structure_before_download()
    print("\n" + "=" * 70)
    print("CONVERTER TEMPLATE")
    print("=" * 70)
    print(create_converter_template())
    
    print("\n" + "=" * 70)
    analyze_synthetic_vs_real()
    
    print("\n=== NEXT STEPS ===")
    print("1. Download real dataset (UTA-RLDD atau UL-DD)")
    print("2. Analyze column structure")
    print("3. Convert ke format pipeline")
    print("4. REMOVE synthetic data (or keep 10% untuk validation)")
    print("5. Retrain dengan:")
    print("   - Natural imbalance (60% alert, 25% drowsy, 15% distracted)")
    print("   - class_weight = {alert:1, drowsy:5, distracted:1}")
    print("   - Prioritize drowsy recall > 95%")
