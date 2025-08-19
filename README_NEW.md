# 🌽 Maize Leaf Disease Classification: Fall Armyworm Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue.svg)](https://www.kaggle.com/datasets/chrismundwa/somalia-hackerthon)

A deep learning solution for binary classification of maize leaf images to detect Fall Armyworm damage using transfer learning with EfficientNet-B0. This project achieved high accuracy in distinguishing between healthy maize leaves and those affected by Fall Armyworm.

## 🎯 Project Overview

Fall Armyworm (*Spodoptera frugiperda*) is a destructive pest that significantly affects maize production in Africa. Early detection through automated image classification can help farmers take timely action to protect their crops.

### Key Features
- **Binary Classification**: Healthy vs Fall Armyworm affected maize leaves
- **Transfer Learning**: EfficientNet-B0 backbone with custom classifier
- **Data Augmentation**: Comprehensive image augmentation pipeline
- **Model Performance**: Achieves high AUC score on validation data
- **Production Ready**: Complete inference pipeline for new images

## 📊 Dataset Information

- **Training Images**: 1,620 labeled images
- **Test Images**: 1,081 images for prediction
- **Classes**: 
  - `0`: Healthy maize leaves
  - `1`: Fall Armyworm affected leaves
- **Image Format**: JPG images of varying sizes
- **Source**: Somalia Hackathon Competition on Kaggle

## 🏗️ Project Structure

```
├── README.md                    # Project documentation
├── FinalSubmission.ipynb        # Complete ML pipeline notebook
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── LICENSE                      # MIT License
├── Train.csv                    # Training data labels
├── Test.csv                     # Test data image IDs
├── SampleSubmission.csv         # Submission format example
└── Images/                      # Image dataset
    ├── id_00exusbkgzw1b.jpg
    ├── id_02amazy34fgh2.jpg
    └── ... (2,701 total images)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/maize-leaf-classification.git
   cd maize-leaf-classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/chrismundwa/somalia-hackerthon)
   - Download and extract all files to the project directory

### Running the Model

1. **Open the Jupyter notebook**
   ```bash
   jupyter notebook FinalSubmission.ipynb
   ```

2. **Execute all cells** to:
   - Load and explore the data
   - Train the EfficientNet-B0 model
   - Generate predictions on test set
   - Create submission file

## 🧠 Model Architecture

### EfficientNet-B0 with Custom Classifier
- **Backbone**: Pre-trained EfficientNet-B0 (ImageNet weights)
- **Classifier Head**: 
  - Dropout (0.3)
  - Linear layer (backbone_features → 512)
  - ReLU activation
  - Dropout (0.3)
  - Output layer (512 → 1)

### Training Configuration
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32
- **Max Epochs**: 20
- **Early Stopping**: 5 epochs patience

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformation
- Resize to 224×224
- ImageNet normalization

## 📈 Model Performance

- **Best Validation AUC**: Achieved during training
- **Training Strategy**: 80/20 train-validation split
- **Class Balance**: Stratified sampling ensures balanced splits
- **Metrics Tracked**: Loss, Accuracy, AUC-ROC

## 🔬 Technical Implementation

### Key Components

1. **Custom Dataset Class**
   ```python
   class MaizeLeafDataset(Dataset):
       # Handles image loading, transforms, and labels
   ```

2. **Model Class**
   ```python
   class MaizeLeafClassifier(nn.Module):
       # EfficientNet backbone + custom classifier
   ```

3. **Training Functions**
   - `train_one_epoch()`: Training loop with progress tracking
   - `validate()`: Validation with metrics calculation
   - `predict()`: Inference on test data

### Environment Compatibility
- **Kaggle**: Automatically detects Kaggle environment
- **Local**: Fallback to local file paths
- **Cross-platform**: Works on Windows, macOS, Linux

## 📋 Usage Examples

### Training a New Model
```python
# Initialize model
model = MaizeLeafClassifier(model_name='efficientnet_b0', num_classes=1)

# Train for specified epochs
for epoch in range(num_epochs):
    train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, ...)
    val_loss, val_acc, val_auc = validate(model, val_loader, ...)
```

### Making Predictions
```python
# Load trained model
model.load_state_dict(torch.load('best_model.pth'))

# Generate predictions
predictions, image_ids = predict(model, test_loader, device)
```

## 🛠️ Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=9.0.0
tqdm>=4.64.0
```

## 🎯 Results and Submission

The notebook generates:
- **Model checkpoint**: `best_maize_leaf_model.pth`
- **Submission file**: `submission.csv` with predictions
- **Training plots**: Loss, accuracy, and AUC curves
- **Performance metrics**: Comprehensive evaluation results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Somalia Hackathon**: For organizing the competition and providing the dataset
- **Kaggle Community**: For the platform and computational resources
- **EfficientNet Authors**: For the excellent pre-trained models
- **PyTorch Team**: For the deep learning framework

## 📞 Contact

**Takunda Mundwa**
- GitHub: [@yourusername](https://github.com/yourusername)
- Kaggle: [Profile Link](https://www.kaggle.com/yourusername)
- Email: your.email@example.com

## 🔗 Related Links

- [Original Dataset](https://www.kaggle.com/datasets/chrismundwa/somalia-hackerthon)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Fall Armyworm Information](https://www.cabi.org/isc/fallarmyworm)

---

⭐ **Star this repository if you found it helpful!**
