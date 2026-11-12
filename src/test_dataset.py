from dataset import PlantDataset
from torch.utils.data import DataLoader
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


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)