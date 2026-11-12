Plant Disease Classification

A multi-class image classification project for identifying plant diseases using Deep Learning and PyTorch.

Project Overview

The goal of this project is to develop a deep learning model capable of classifying plant leaf images into 38 different categories, including healthy plants and various diseases.

The project follows a structured machine learning workflow, from data exploration and preprocessing to model training and evaluation.

Dataset

This project uses a PlantVillage-style dataset containing images of healthy and diseased plant leaves.

- Original images: 54,305
- Classes: 38
- Image dimensions: 256 × 256
- Color space: RGB

Data preprocessing includes duplicate detection, data validation, and stratified train/validation/test splitting.

After removing duplicate images, the dataset contains 54,284 unique images.

Split| Images
Training| 37,998
Validation| 8,143
Testing| 8,143
Total| 54,284

Note: The dataset is not included in this repository.

Project Structure

plant-disease-classification/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── prepare_data.py
│   ├── test_dataset.py
│   └── train.py
│
├── .gitignore
└── README.md

Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn

Current Progress

- [x] Dataset exploration and validation
- [x] Duplicate image detection and removal
- [x] Stratified train/validation/test splitting
- [x] PyTorch Dataset implementation
- [ ] Data augmentation
- [ ] Model development and training
- [ ] Model evaluation
- [ ] Performance analysis and visualization

Future Work

- Implement and compare deep learning architectures.
- Apply data augmentation techniques.
- Evaluate model performance using appropriate classification metrics.
- Analyze class-wise performance and misclassifications.
- Explore transfer learning approaches.

Author

Daniyal Bayat

M.Sc. Student in Automotive Engineering | Machine Learning & Computer Vision

"GitHub Profile" (https://github.com/daniyal-bayat)