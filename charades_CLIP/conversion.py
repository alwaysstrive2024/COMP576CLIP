#!/usr/bin/env python3
import csv, json, os

TRAIN_CSV = "charades/annotations/Charades_v1_train.csv"
VAL_CSV   = "charades/annotations/Charades_v1_test.csv"
OUT_DIR   = "charades/annotations"
OUT_TRAIN = os.path.join(OUT_DIR, "charades_sta_train.json")
OUT_VAL   = os.path.join(OUT_DIR, "charades_sta_val.json")

def csv_to_sta(csv_path):
    recs = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["id"]
            query = row["descriptions"].strip() if row.get("descriptions") else ""
            actions = row.get("actions") or ""
            # split into segments by semicolon
            segments = [seg.strip() for seg in actions.split(";") if seg.strip()]
            for seg in segments:
                parts = seg.split()
                if len(parts) != 3:
                    # malformed segment? skip it
                    continue
                _, start_s, end_s = parts
                try:
                    start, end = float(start_s), float(end_s)
                except ValueError:
                    continue
                recs.append({
                    "video_id": vid,
                    "query":     query,
                    "start_sec": start,
                    "end_sec":   end
                })
    return recs

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    train_recs = csv_to_sta(TRAIN_CSV)
    val_recs   = csv_to_sta(VAL_CSV)
    print(f"Train segments: {len(train_recs)},  Val segments: {len(val_recs)}")
    merged_recs = train_recs + val_recs
    OUT_MERGED = os.path.join(OUT_DIR, "charades_sta.json")
    with open(OUT_MERGED, "w") as f:
        json.dump(merged_recs, f, indent=2)
    print("Wrote merged file:", OUT_MERGED)


