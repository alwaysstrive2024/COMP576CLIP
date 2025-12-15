#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import csv
import torch
import clip
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def infer_top3_segments(video_path, query, model, preprocess, device,
                        sample_rate=5, window_size=2.0, stride=1.0):
    """
    Perform zero-shot inference for a single video-query pair.
    Returns the top-3 candidate segments after boundary expansion.
    """
    # Encode text query
    text_tokens = clip.tokenize([query], truncate=True).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens).float()
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    # Sample frames and encode images
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps, img_feats = [], []
    for idx in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(pil).float()
            feat /= feat.norm(dim=-1, keepdim=True)
        timestamps.append(t)
        img_feats.append(feat.cpu().numpy().squeeze())
    cap.release()
    if not img_feats:
        return None

    timestamps = np.array(timestamps)     # shape: [T]
    img_feats = np.stack(img_feats, 0)    # shape: [T, D]
    text_vec = text_feat.cpu().numpy().squeeze()  # shape: [D]
    T = len(timestamps)

    # Sliding window based on time rather than frame step
    raw_cands = []
    i = 0
    while i < T:
        start_time = timestamps[i]
        end_limit = start_time + window_size
        # find last index j where timestamps[j] <= end_limit
        j = np.searchsorted(timestamps, end_limit, side='right') - 1
        if j >= i:
            seg_feats = img_feats[i:j+1]
            avg_feat = seg_feats.mean(axis=0)
            avg_feat /= np.linalg.norm(avg_feat)
            score = float(np.dot(avg_feat, text_vec))
            raw_cands.append({'i': i, 'j': j, 'score': score})
        # advance by stride seconds
        next_time = start_time + stride
        i = np.searchsorted(timestamps, next_time, side='left')

    if not raw_cands:
        return None

    # sort and keep top-3
    raw_cands.sort(key=lambda x: x['score'], reverse=True)
    top3_raw = raw_cands[:3]

    # boundary expansion around peak-to-valley
    refined = []
    for cand in top3_raw:
        i, j, score = cand['i'], cand['j'], cand['score']
        sims = (img_feats[i:j+1] @ text_vec)  # dot products
        # normalized similarity sequence
        norm_sims = sims / np.linalg.norm(sims)
        # find peak relative index
        peak_rel = int(np.argmax(norm_sims))
        peak_idx = i + peak_rel

        # scan left to valley
        l = peak_idx
        while l > i and norm_sims[l-i] >= norm_sims[l-i-1]:
            l -= 1
        # scan right to valley
        r = peak_idx
        while r < j and norm_sims[r-i] >= norm_sims[r-i+1]:
            r += 1

        refined.append({
            'start': timestamps[l],
            'end':   timestamps[r],
            'score': score
        })

    # final sort and return
    refined.sort(key=lambda x: x['score'], reverse=True)
    return refined[:3]


def compute_iou(pred, gt_start, gt_end):
    """
    Compute temporal Intersection-over-Union between a predicted segment and ground truth.
    """
    inter = max(0, min(pred['end'], gt_end) - max(pred['start'], gt_start))
    union = max(pred['end'], gt_end) - min(pred['start'], gt_start) + 1e-6
    return inter / union

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load annotations
    with open(args.ann, 'r') as f:
        annotations = json.load(f)

    # Prepare output CSV directory
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    csvfile = open(args.output_csv, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow([
        'video_id', 'query',
        'cand1_start', 'cand1_end', 'cand1_score',
        'cand2_start', 'cand2_end', 'cand2_score',
        'cand3_start', 'cand3_end', 'cand3_score',
        'gt_start', 'gt_end', 'pred1_iou', 'running_mIoU'
    ])

    running_iou_sum = 0.0
    success = 0

    pbar = tqdm(total=len(annotations), desc="Zero-Shot Inference")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_rec = {
            executor.submit(
                infer_top3_segments,
                os.path.join(args.video_root, f"{rec['video_id']}.mp4"),
                rec['query'], model, preprocess, device,
                args.sample_rate, args.window_size, args.stride
            ): rec for rec in annotations
        }

        for future in as_completed(future_to_rec):
            rec = future_to_rec[future]
            pbar.update(1)
            try:
                result = future.result()
            except Exception as e:
                tqdm.write(f"[ERROR] {rec['video_id']}: {e}")
                continue

            if result is None:
                continue

            vid    = rec['video_id']
            query  = rec['query']
            gt_s   = rec['start_sec']
            gt_e   = rec['end_sec']
            top3   = result
            iou1   = compute_iou(top3[0], gt_s, gt_e)

            success += 1
            running_iou_sum += iou1
            current_miou = running_iou_sum / success
            pbar.set_postfix({'mIoU': f"{current_miou:.3f}"})

            writer.writerow([
                vid, query,
                f"{top3[0]['start']:.2f}", f"{top3[0]['end']:.2f}", f"{top3[0]['score']:.4f}",
                f"{top3[1]['start']:.2f}", f"{top3[1]['end']:.2f}", f"{top3[1]['score']:.4f}",
                f"{top3[2]['start']:.2f}", f"{top3[2]['end']:.2f}", f"{top3[2]['score']:.4f}",
                f"{gt_s:.2f}", f"{gt_e:.2f}",
                f"{iou1:.3f}", f"{current_miou:.3f}"
            ])

    pbar.close()
    csvfile.close()

    # # Save the model after inference
    # model_save_path = args.output_ckpt  # Path where you want to save the model
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists
    # torch.save(model.state_dict(), model_save_path)
    # print(f"[INFO] Model saved to {model_save_path}")

    print(f"[DONE] Results saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Charades-STA Zero-Shot Inference with CLIP"
    )
    parser.add_argument("--ann",        required=True,
                        help="Path to Charades-STA annotation JSON")
    parser.add_argument("--video_root", required=True,
                        help="Directory containing videos named <video_id>.mp4")
    parser.add_argument("--output_csv", required=True,
                        help="Path to output CSV file")
    # parser.add_argument("--output_ckpt", required=True,
    #                     help="Path to save the model checkpoint")
    parser.add_argument("--sample_rate", type=int,   default=5,
                        help="Frame sampling interval (frames)")
    parser.add_argument("--window_size", type=float, default=2.0,
                        help="Sliding window duration (seconds)")
    parser.add_argument("--stride",      type=float, default=1.0,
                        help="Sliding window stride (seconds)")
    parser.add_argument("--workers",     type=int,   default=15,
                        help="Number of parallel worker threads")
    args = parser.parse_args()

    main(args)


# How to run:
# python -u train.py 
#   --ann   scratch/annotations/charades_sta.json 
#   --video_root  scratch/videos 
#   --output_csv   charades_zeroshot_results.csv 
#   --output_ckpt model.pt
#   --sample_rate  5 
#   --window_size  2.0 
#   --stride       1.0
#   --workers       15
# python -u train.py --ann   /scratch/charades/annotations/charades_sta.json --video_root  /scratch/charades/videos --output_csv   charades_zeroshot_results.csv --output_csv   charades_zeroshot_results.csv --output_ckpt model.pt --sample_rate  5 --window_size  2.0 --stride       1.0 --workers       15