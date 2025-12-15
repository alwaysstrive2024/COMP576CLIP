
import json, random, shutil, os

SRC_ANN = "path/to/charades_sta_train.json"
SRC_VID = "path/to/Charades_videos"
TMP_DIR = "./tmp_smoke"
N_SAMPLE = 5  

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(f"{TMP_DIR}/videos", exist_ok=True)

anns = json.load(open(SRC_ANN, 'r'))
sampled = random.sample(anns, N_SAMPLE)
with open(f"{TMP_DIR}/small_train.json", 'w') as f:
    json.dump(sampled, f, indent=2)

for e in sampled:
    vid = e['video_id'] + ".mp4"
    shutil.copy(os.path.join(SRC_VID, vid), f"{TMP_DIR}/videos/{vid}")

# print("已准备好小规模测试集，运行指令：")
print(f"python charades_sta_train.py \\\n"
      f"  --train_ann   {TMP_DIR}/small_train.json \\\n"
      f"  --val_ann     {TMP_DIR}/small_train.json \\\n"
      f"  --video_root  {TMP_DIR}/videos \\\n"
      f"  --batch_size  1 \\\n"
      f"  --epochs      1 \\\n"
      f"  --lr          1e-4 \\\n"
      f"  --output_ckpt {TMP_DIR}/smoke.pt")

