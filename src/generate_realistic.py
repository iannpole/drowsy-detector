import csv
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def generate_realistic_imbalanced(subject_id, rows=2000, seed=None):
    """Generate realistic imbalanced dataset matching real-world distribution."""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    baseline = 0.30 + np.random.normal(0, 0.02)
    data = []
    
    # Shuffle indices untuk realistic sequencing (bukan pure alert → drowsy → distracted)
    indices = list(range(rows))
    random.shuffle(indices)
    
    alert_count = int(rows * 0.60)
    drowsy_count = int(rows * 0.25)
    distracted_count = rows - alert_count - drowsy_count
    
    labels = ["alert"] * alert_count + ["drowsy"] * drowsy_count + ["distracted"] * distracted_count
    labels = [labels[i] for i in indices]  # Shuffle labels
    
    for i, label in enumerate(labels):
        timestamp = i * 0.5
        
        if label == "alert":
            ear = baseline + np.random.normal(0, 0.012)
            mar = 0.08 + np.random.normal(0, 0.008)
            perclos = 0.05 if ear > baseline * 0.7 else np.random.uniform(0, 0.08)
            pitch = np.random.normal(1.5, 2.5)
            yaw = np.random.normal(0, 3)
            blink_rate = np.random.uniform(17, 23)
            max_closed = np.random.uniform(0, 0.15)
        
        elif label == "drowsy":
            ear = baseline * 0.62 + np.random.normal(0, 0.018)
            mar = 0.055 + np.random.normal(0, 0.012)
            perclos = np.random.uniform(0.35, 0.75)
            pitch = np.random.normal(9, 5.5)
            yaw = np.random.normal(0, 5)
            blink_rate = np.random.uniform(6, 12)
            max_closed = np.random.uniform(0.8, 2.5)
        
        else:  # distracted
            ear = baseline + np.random.normal(0, 0.012)
            mar = 0.08 + np.random.normal(0, 0.008)
            perclos = 0.05 if ear > baseline * 0.7 else np.random.uniform(0, 0.06)
            pitch = np.random.normal(np.random.choice([-12, 12]), 7)
            yaw = np.random.normal(np.random.choice([-28, 28]), 8)
            blink_rate = np.random.uniform(16, 24)
            max_closed = np.random.uniform(0, 0.2)
        
        data.append({
            "timestamp": f"{timestamp:.3f}",
            "subject": subject_id,
            "ear_mean": f"{max(0.01, min(0.99, ear)):.6f}",
            "ear_std": f"{abs(np.random.normal(0, 0.008)):.6f}",
            "ear_norm": f"{max(0.1, min(4.5, ear / baseline)):.6f}",
            "mar_mean": f"{max(0, min(5, mar)):.6f}",
            "perclos": f"{max(0, min(1, perclos)):.6f}",
            "blink_rate": f"{max(0, min(300, blink_rate)):.6f}",
            "max_closed_sec": f"{max(0, min(3, max_closed)):.6f}",
            "pitch_mean": f"{max(-180, min(180, pitch)):.6f}",
            "pitch_std": f"{abs(np.random.normal(0, 1.5)):.6f}",
            "yaw_mean": f"{max(-180, min(180, yaw)):.6f}",
            "yaw_std": f"{abs(np.random.normal(0, 2.5)):.6f}",
            "label": label,
        })
    
    return data


def create_realistic_dataset():
    """Create 5 subjects × 2000 rows = 10k rows dengan natural distribution."""
    DATA_DIR.mkdir(exist_ok=True)
    
    cols = [
        "timestamp", "subject", "ear_mean", "ear_std", "ear_norm",
        "mar_mean", "perclos", "blink_rate", "max_closed_sec",
        "pitch_mean", "pitch_std", "yaw_mean", "yaw_std", "label"
    ]
    
    print("=== Generate Realistic Imbalanced Dataset ===\n")
    print("Distribution: Alert 60% | Drowsy 25% | Distracted 15%")
    print("Total: 5 subjects × 2000 rows = 10,000 rows (natural imbalance)\n")
    
    all_stats = {"alert": 0, "drowsy": 0, "distracted": 0}
    
    for subj in range(1, 6):
        data = generate_realistic_imbalanced(f"real_{subj}", rows=2000, seed=subj * 1000)
        filename = DATA_DIR / f"real_{subj}_balanced.csv"
        
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(data)
        
        stats = {}
        for row in data:
            label = row["label"]
            stats[label] = stats.get(label, 0) + 1
            all_stats[label] += 1
        
        pct = f"Alert: {stats.get('alert', 0):4d} (60%) | Drowsy: {stats.get('drowsy', 0):4d} (25%) | Distracted: {stats.get('distracted', 0):4d} (15%)"
        print(f"✓ real_{subj}_balanced.csv: 2000 rows → {pct}")
    
    print(f"\n=== TOTAL ===")
    print(f"Alert:       {all_stats['alert']:5d} ({all_stats['alert']*100//10000}%)")
    print(f"Drowsy:      {all_stats['drowsy']:5d} ({all_stats['drowsy']*100//10000}%)")
    print(f"Distracted:  {all_stats['distracted']:5d} ({all_stats['distracted']*100//10000}%)")
    print(f"TOTAL:       {sum(all_stats.values()):5d} rows")


if __name__ == "__main__":
    create_realistic_dataset()
