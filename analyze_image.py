import os
from pathlib import Path

base_path = Path("artifacts/data_ingestion/Chicken-fecal-images")
class_dirs = ["Coccidiosis", "Healthy"]
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.tif', '.tiff', '.gif')

total_images = 0
total_files = 0
zero_byte_files = 0
unrecognized_files = 0

print("=== IMAGE DATASET ANALYSIS ===\n")

for class_dir in class_dirs:
    folder = os.path.join(base_path, class_dir)
    if not os.path.exists(folder):
        print(f"Folder missing: {folder}")
        continue

    images = []
    non_images = []
    zeros = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            total_files += 1
            if f.lower().endswith(valid_exts):
                images.append(f)
                if os.path.getsize(path) == 0:
                    zeros.append(f)
                    zero_byte_files += 1
            else:
                non_images.append(f)
                unrecognized_files += 1

    print(f"Class: {class_dir}")
    print(f"  Total files: {len(os.listdir(folder))}")
    print(f"  Image files: {len(images)}")
    print(f"  Non-image files: {len(non_images)}")
    print(f"  Zero-byte images: {len(zeros)}")
    if len(non_images) > 0:
        print(f"    Example non-image: {non_images[:3]}")
    if len(zeros) > 0:
        print(f"    Example zero-byte: {zeros[:3]}")
    print()
    total_images += len(images)

print("=== SUMMARY ===")
print(f"Total image files: {total_images}")
print(f"Total files (all types): {total_files}")
print(f"Total zero-byte images: {zero_byte_files}")
print(f"Total unrecognized files: {unrecognized_files}")

# Optional: print a few sample file names from each class
for class_dir in class_dirs:
    folder = os.path.join(base_path, class_dir)
    print(f"\nSample files in {class_dir}:")
    print(os.listdir(folder)[:5])