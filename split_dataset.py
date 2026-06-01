import os
import random
import shutil

# ===== paths =====
img_dir = "dataset/images"
label_dir = "dataset/labels"

train_img = "dataset/images/train"
val_img = "dataset/images/val"

train_label = "dataset/labels/train"
val_label = "dataset/labels/val"

os.makedirs(train_img, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(train_label, exist_ok=True)
os.makedirs(val_label, exist_ok=True)

# only real images
images = [f for f in os.listdir(img_dir)
          if f.lower().endswith(('.jpg','.jpeg','.png'))]

print("Total images:", len(images))

random.shuffle(images)

split = int(0.8 * len(images))

for i, img in enumerate(images):
    name = os.path.splitext(img)[0]

    img_src = os.path.join(img_dir, img)
    label_src = os.path.join(label_dir, name + ".txt")

    if i < split:
        shutil.copy2(img_src, os.path.join(train_img, img))
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(train_label, name + ".txt"))
    else:
        shutil.copy2(img_src, os.path.join(val_img, img))
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(val_label, name + ".txt"))

print("TRAIN/VAL SPLIT DONE ✔")