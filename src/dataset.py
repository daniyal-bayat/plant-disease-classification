import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, dataloader
from pathlib import Path
import numpy as np

# Making dataset class
class PlantDataset(Dataset):
    TRAIN_CSV = Path("PlantDiseaseClassification/data/splits/train.csv")
    VAL_CSV = Path("PlantDiseaseClassification/data/splits/validation.csv")
    TEST_CSV = Path("PlantDiseaseClassification/data/splits/test.csv")

    def __init__(self, csv_file, data_dir):
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_dir / row["image_path"]
        label = int(row["class_id"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image)
        image = image.float() / 255.0
        return image, label

