from dataset import PlantDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_dataset = PlantDataset(
    csv_file=PlantDataset.TRAIN_CSV,
    data_dir="PlantDiseaseClassification/data/color"
)

print("Dataset length:", len(train_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min:", image.min())
print("Image max:", image.max())

print("Label:", label)
print("Label type:", type(label))

validation_dataset = PlantDataset(
    csv_file=PlantDataset.VAL_CSV,
    data_dir="PlantDiseaseClassification/data/color"
)

test_dataset = PlantDataset(
    csv_file=PlantDataset.TEST_CSV,
    data_dir="PlantDiseaseClassification/data/color"
)

# data loader testing
# train-loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# val-loader
validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# test-loader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

images, labels = next(iter(validation_loader))

print(images.shape)
print(labels.shape)

images, labels = next(iter(test_loader))

print(images.shape)
print(labels.shape)

# visual inspection with matplotlib
# Get one batch from DataLoader
images, labels = next(iter(train_loader))

# Get class names
class_names = sorted(train_dataset.df["label"].unique())

# Create a figure with 2 rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Display 8 images
for i, ax in enumerate(axes.flat):

    # Convert image from CHW to HWC
    image = images[i].permute(1, 2, 0)

    # Get the actual class name
    label = labels[i].item()
    class_name = class_names[label]

    # Display image
    ax.imshow(image)

    # Set title
    ax.set_title(class_name, fontsize=8)

    # Hide axis
    ax.axis("off")

plt.tight_layout()
plt.show()

