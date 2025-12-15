import os, pathlib, random, math, json, time 
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import clip

def load_charades_sta(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segment, sentence = line.split('##', 1)
            vid, start, end = segment.split()
            records.append({
                'video_id':   vid,
                'start_time': float(start),
                'end_time':   float(end),
                'sentence':   sentence
            })
    return pd.DataFrame(records)

df_test  = load_charades_sta('./charades/charades_sta_test.txt')
df_train = load_charades_sta('./charades/charades_sta_train.txt')

video_dir = "/charades/videos"
video_ids_local = [p.stem for p in pathlib.Path(video_dir).glob("*")]
print(video_ids_local)
# assert len(video_ids_local) == 10


eval_df = df_train[df_train.video_id.isin(video_ids_local)].reset_index(drop=True)
assert len(eval_df) > 0, "test.txt 中没有匹配这 10 个视频的标注行！"

# ---------- CLIP ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- 推理函数（你的实现） ----------
@torch.no_grad()
def infer_top3_segments(video_path, query,
                        sample_rate=20, extend=4.0, topk=1):

    # --- encode text ---
    text_tokens = clip.tokenize([query], truncate=True).to(device)
    text_feat   = model.encode_text(text_tokens).float()
    text_feat  /= text_feat.norm(dim=-1, keepdim=True)
    text_vec    = text_feat.cpu().numpy().squeeze()

    # --- sample frames & encode image ---
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 1.0
    fcnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ts, feats = [], []
    for idx in range(0, fcnt, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        ts.append(idx / fps)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
        feat = model.encode_image(pil).float()
        feat /= feat.norm(dim=-1, keepdim=True)
        feats.append(feat.cpu().numpy().squeeze())
    cap.release()
    if not feats:
        return []

    ts     = np.array(ts)            # [T]
    feats  = np.stack(feats, 0)      # [T, D]
    sims   = feats @ text_vec        # [T]

    # --- Find Similarity Frames ---
    peak_i   = int(np.argmax(sims))
    peak_t   = ts[peak_i]
    peak_sim = float(sims[peak_i])

    # --- extend seconds left and right ---
    vid_dur = fcnt / fps
    seg_st  = max(0.0, peak_t - extend)
    seg_ed  = min(vid_dur, peak_t + extend)

    return [{"start": seg_st, "end": seg_ed, "score": peak_sim}]

# ---------- IoU & Recall ----------
def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0

hits = 0
# thresholds at which to compute Recall@1
iou_thresholds = [0.3, 0.5, 0.7]

# prepare storage for per-sample results
records = []

# loop through all eval samples
for idx, row in tqdm(eval_df.iterrows(),
                     total=len(eval_df),
                     desc="evaluating"):
    vid, g_st, g_ed, sent = (
        row.video_id, row.start_time, row.end_time, row.sentence
    )
    vpath = os.path.join(video_dir, f"{vid}.mp4")
    preds = infer_top3_segments(vpath, sent)
    if not preds:
        # if no prediction, IoU = 0, no hits at any threshold
        iou_val = 0.0
        hits = {thr: 0 for thr in iou_thresholds}
    else:
        top1 = preds[0]
        iou_val = iou((top1["start"], top1["end"]), (g_st, g_ed))
        # for each threshold, record whether this sample is a “hit”
        hits = {thr: int(iou_val >= thr) for thr in iou_thresholds}

    # store one record per sample
    rec = {
        "video_id": vid,
        "gt_start": g_st,
        "gt_end": g_ed,
        "pred_start": preds[0]["start"] if preds else None,
        "pred_end":   preds[0]["end"]   if preds else None,
        "iou": iou_val,
    }
    # add hit flags
    for thr in iou_thresholds:
        rec[f"hit@{thr}"] = hits[thr]
    records.append(rec)

# convert to DataFrame
results_df = pd.DataFrame(records)

# compute Recall@1 for each threshold
recalls = {}
for thr in iou_thresholds:
    recalls[thr] = results_df[f"hit@{thr}"].mean()

# print them out
for thr, r in recalls.items():
    print(f"Recall@1 (IoU≥{thr}): {r:.3f}")

# save full per-sample results (and you can later aggregate yourself)
results_df.to_csv("results.csv", index=False)
print("All per-sample IoUs and hit flags saved to results.csv")