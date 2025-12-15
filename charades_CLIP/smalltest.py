#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, random, shutil, os
# 复制少量注释与视频到临时目录，然后调用上面的训练脚本一小轮

# 配置
SRC_ANN = "path/to/charades_sta_train.json"
SRC_VID = "path/to/Charades_videos"
TMP_DIR = "./tmp_smoke"
N_SAMPLE = 5  # 随机抽 5 条做测试

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(f"{TMP_DIR}/videos", exist_ok=True)

# 加载并抽样注释
anns = json.load(open(SRC_ANN, 'r'))
sampled = random.sample(anns, N_SAMPLE)
with open(f"{TMP_DIR}/small_train.json", 'w') as f:
    json.dump(sampled, f, indent=2)

# 复制对应视频
for e in sampled:
    vid = e['video_id'] + ".mp4"
    shutil.copy(os.path.join(SRC_VID, vid), f"{TMP_DIR}/videos/{vid}")

print("已准备好小规模测试集，运行指令：")
print(f"python charades_sta_train.py \\\n"
      f"  --train_ann   {TMP_DIR}/small_train.json \\\n"
      f"  --val_ann     {TMP_DIR}/small_train.json \\\n"
      f"  --video_root  {TMP_DIR}/videos \\\n"
      f"  --batch_size  1 \\\n"
      f"  --epochs      1 \\\n"
      f"  --lr          1e-4 \\\n"
      f"  --output_ckpt {TMP_DIR}/smoke.pt")

# python charades_sta_train.py \
#   --train_ann   path/to/charades_sta_train.json \
#   --val_ann     path/to/charades_sta_val.json \
#   --video_root  path/to/Charades_videos/ \
#   --batch_size  2 \
#   --epochs      10 \
#   --lr          1e-4 \
#   --output_ckpt best_charades.pt
