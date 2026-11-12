from pathlib import Path
from PIL import Image
from collections import Counter
from collections import defaultdict
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset path
DATA_DIR = Path("PlantDiseaseClassification/data/color")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Class names, Picture numbers in each class, Number of classes
num_classes = 0
total = 0
class_names = []
image_counts = []
print("subfolder names:")
for folder in DATA_DIR.iterdir():
    if folder.is_dir():
        num_classes = num_classes + 1
        image_count = 0
        extensions = ["*.jpg", "*.jpeg", "*.png"]
        for extension in extensions:
            image_count = image_count + len(list(folder.glob(extension)))
        total = total + (image_count)
        class_names.append(folder.name)
        image_counts.append(image_count)
        print(folder.name, ":", image_count, "pictures")
print("\n\n\nnumber of classes : ", num_classes)
print(f"total pictures count: {total}")

# Dataset quality check
total_images = 0
healthy_images = 0
broken_images = 0

image_sizes = Counter()
image_modes = Counter()

for image_path in DATA_DIR.rglob("*"):

    # Only process image files
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    total_images += 1

    try:
        # First, verify the image file
        with Image.open(image_path) as image:
            image.verify()

        # Reopen the file to inspect and load its pixel data
        with Image.open(image_path) as image:
            image.load()
            image_sizes[image.size] += 1
            image_modes[image.mode] += 1

        healthy_images += 1

    except Exception as error:
        broken_images += 1
        print(f"Broken image: {image_path}")
        print(f"Reason: {error}")

# Printing results
print("\n========== DATASET REPORT ==========")
print(f"Total image files: {total_images}")
print(f"Healthy images: {healthy_images}")
print(f"Broken images: {broken_images}")

print("\nTop 5 image dimensions:")
for size, count in image_sizes.most_common(5):
    print(f"{size}: {count} images")

print("\nImage color modes:")
for mode, count in image_modes.most_common():
    print(f"{mode}: {count} images")

# Finding duplicate images
hash_to_paths = defaultdict(list)

# Finding images hash
for image_path in DATA_DIR.rglob("*"):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    with image_path.open("rb") as file:
        file_hash = hashlib.sha256(file.read()).hexdigest()

    hash_to_paths[file_hash].append(image_path)

# Finding duplicate images
duplicate_groups = [
    paths for paths in hash_to_paths.values()
    if len(paths) > 1
]

for index, paths in enumerate(duplicate_groups, start=1):
    class_names = {path.parent.name for path in paths}

    print(f"\nGroup {index}")
    print(f"Number of files: {len(paths)}")
    print(f"Classes: {class_names}")

    for path in paths:
        print(path)

duplicate_files = sum(
    len(paths) - 1 for paths in duplicate_groups
)

# Printing results
print(f"Total duplicate groups: {len(duplicate_groups)}")
print(f"Extra duplicate files: {duplicate_files}")

for paths in duplicate_groups[:10]:
    print("\nDuplicate group:")
    for path in paths:
        print(path)

# Class n
# ames
class_names = sorted (
    fold.name for fold in DATA_DIR.iterdir() if fold.is_dir()
)

print(f"number of classes:{len(class_names)}\n")
print("first 5 classes:")
print("\n".join(class_names[:5]))
print()

# Class ids
class_to_id = {
    name: index for index, name in enumerate(class_names)
}

print("class_ids:")
for name, index in class_to_id.items():
    print(f"{name}: {index}")

# Making dataframe
data = []

for image_path in DATA_DIR.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    label = image_path.parent.name
    class_id = class_to_id[label]
    relative_path = image_path.relative_to(DATA_DIR)

    data.append({
        "image_path": str(relative_path),
        "label": label,
        "class_id": class_id
    })

df = pd.DataFrame(data)

print("\nDataset Manifest:")
print(df.head())

print(f"\nTotal images: {len(df)}")
print(f"Number of classes: {df["label"].nunique()}")

print("\nImages per class:")
print(df["label"].value_counts())

# Saving dataframe in CSV
MANIFEST_PATH = DATA_DIR.parent / "manifest.csv"

df.to_csv(MANIFEST_PATH, index=False)

print(f"\nManifest saved to: {MANIFEST_PATH}")

# Deleting duplicate files from CSV
hashes = []

for relative_path in df["image_path"]:
    image_path = DATA_DIR / relative_path

    with image_path.open("rb") as file:
        image_hash = hashlib.sha256(file.read()).hexdigest()

    hashes.append(image_hash)

df["image_hash"] = hashes

df_clean = df.drop_duplicates(
    subset="image_hash",
    keep="first"
).copy()

print("Original images:", len(df))
print("Images after deduplication:", len(df_clean))
print("Removed duplicate copies:", len(df) - len(df_clean))

# Stratified spliting data (train - test - validation)
# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=42,
    stratify=df_clean["label"]
)

# Second split: divide the temporary set equally
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("Total:", len(train_df) + len(val_df) + len(test_df))

print("Train classes:", train_df["label"].nunique())
print("Validation classes:", val_df["label"].nunique())
print("Test classes:", test_df["label"].nunique())

print("\nTrain class distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["label"].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df["label"].value_counts().sort_index())

# Removeing the temporary hash column
train_df = train_df.drop(columns=["image_hash"], errors="ignore")
val_df = val_df.drop(columns=["image_hash"], errors="ignore")
test_df = test_df.drop(columns=["image_hash"], errors="ignore")

# Saving the final manifests
SPLIT_DIR = DATA_DIR.parent / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
val_df.to_csv(SPLIT_DIR / "validation.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

print("Manifests saved successfully!")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

print("Train file exists:", (SPLIT_DIR / "train.csv").exists())
print("Validation file exists:", (SPLIT_DIR / "validation.csv").exists())
print("Test file exists:", (SPLIT_DIR / "test.csv").exists())