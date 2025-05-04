import os
from collections import Counter

def count_class_ids(label_dir):
    class_counter = Counter()
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    if line.strip() == "":
                        continue
                    class_id = int(line.split()[0])
                    class_counter[class_id] += 1
    return class_counter

# Path label original dan baru
original_label_path = "dataset/train/labels"
augmented_label_path = "dataset-new/train/labels"

# Hitung distribusi class ID di masing-masing dataset
original_counts = count_class_ids(original_label_path)
augmented_counts = count_class_ids(augmented_label_path)

# Tampilkan hasil
print("📊 Distribusi Class ID di Dataset Original:")
for cid in sorted(original_counts):
    print(f"Class {cid}: {original_counts[cid]} labels")

print("\n📊 Distribusi Class ID di Dataset Augmented:")
for cid in sorted(augmented_counts):
    print(f"Class {cid}: {augmented_counts[cid]} labels")

# Perbandingan visual
print("\n🔍 Catatan:")
for cid in range(4):
    o = original_counts.get(cid, 0)
    a = augmented_counts.get(cid, 0)
    if o > 0 and a > 0:
        print(f"✅ Class ID {cid} muncul di kedua dataset.")
    elif o > 0 and a == 0:
        print(f"⚠️ Class ID {cid} hanya ada di dataset original.")
    elif o == 0 and a > 0:
        print(f"⚠️ Class ID {cid} hanya ada di dataset augmented.")
