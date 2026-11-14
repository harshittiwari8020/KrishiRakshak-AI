# KrishiRakshak AI — Advanced Crop Disease Detection

A local Streamlit application that classifies crop-leaf images using transfer learning (EfficientNet-B0), shows the top 3 class scores, stores prediction history in SQLite, and provides bilingual (English/Hindi) general next-step guidance.

## What is included

- `app.py` — farmer-facing web interface.
- `download_dataset.py` — downloads and exports PlantVillage images into ImageFolder folders.
- `train.py` — trains EfficientNet-B0, validates during training, saves the best model, and evaluates on the supplied test split.
- `requirements.txt` — Python dependencies.
- `models/`, `data/`, `outputs/` — generated model, dataset and reports (not bundled).

## Dataset

The project uses the PlantVillage colour-image benchmark: approximately 54,000 images across 38 crop/disease classes. The dataset is not included in this ZIP because it is roughly 0.8 GB or more and has its own source terms. It is downloaded to your PC when you run the downloader.

Dataset reference: TensorFlow Datasets documents 54,303 images in 38 classes and an approximately 828 MiB download: https://www.tensorflow.org/datasets/catalog/plant_village
Hugging Face dataset card: https://huggingface.co/datasets/mohanty/PlantVillage

Please review the dataset's current terms and cite the original PlantVillage paper in academic submissions:
Mohanty, Hughes & Salathé (2016), "Using Deep Learning for Image-Based Plant Disease Detection", Frontiers in Plant Science, DOI: 10.3389/fpls.2016.01419.

**Important limitation:** PlantVillage images were largely captured under controlled backgrounds. Real Indian farm photos can differ in lighting, camera quality, crop varieties and multiple simultaneous problems. A strong benchmark score does not guarantee field accuracy. For a serious India-focused version, collect and label local field images (with consent and expert verification) and evaluate on a separate field-photo test set.

## Recommended computer setup

- Windows 10/11, 64-bit.
- Python **3.11** recommended (avoid Python 3.14 for this stack until all dependencies support it).
- 8 GB RAM minimum; 16 GB recommended.
- At least 5 GB free disk space for environment, dataset, model and cache.
- NVIDIA GPU helps, but CPU training is possible and will take longer.

## Step-by-step Windows setup

Open **PowerShell** in the extracted project folder.

### 1. Check Python

```powershell
py -3.11 --version
```

If that command is not found, install Python 3.11 (64-bit) from https://www.python.org/downloads/ and enable **Add python.exe to PATH**. Close and reopen PowerShell.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run this once in that terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install libraries

```powershell
pip install -r requirements.txt
```

PyTorch installation can vary by GPU/CUDA. If `pip install -r requirements.txt` fails on torch/torchvision, use the official selector at https://pytorch.org/get-started/locally/ and install the matching pair, then run the requirements command again.

### 4. Download the dataset

```powershell
python download_dataset.py
```

This downloads the PlantVillage dataset and exports it under `data\plantvillage`. It needs an internet connection and substantial disk space. The first run can take a while. If Hugging Face asks for access/login, follow the dataset page's current instructions.

### 5. Train the model

Start with 3 epochs to verify that everything works:
```powershell
python train.py --epochs 3 --batch-size 16
```

Then run a longer training:
```powershell
python train.py --epochs 10 --batch-size 16
```

If your computer runs out of memory, reduce batch size:
```powershell
python train.py --epochs 5 --batch-size 8
```

The best model is saved to `models\crop_disease_model.pth`. Test metrics are written to `outputs\test_classification_report.json` when the test split is available. Training time depends heavily on CPU/GPU and disk speed.

### 6. Start the application

```powershell
streamlit run app.py
```

A browser tab should open, usually at `http://localhost:8501`. Upload a clear JPG/PNG leaf photo and click **Analyse leaf**.

### 7. Every time you use the project later

```powershell
cd path\to\Crop_Disease_Detection_Advanced
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Troubleshooting

- **`py -3.11` not found:** install Python 3.11 64-bit and reopen terminal.
- **`streamlit` not recognized:** activate `.venv`, then run `python -m streamlit run app.py`.
- **No trained model:** run `python train.py` after dataset download.
- **Out of memory:** use `--batch-size 8` or `4`; close other applications.
- **Very slow training:** expected on CPU. Train a small test run first; use a CUDA-enabled PyTorch build if you have a supported NVIDIA GPU.
- **Dataset download error:** check internet, dataset access terms and Hugging Face status; rerun `python download_dataset.py`.
- **Windows DataLoader hangs:** keep `--workers 0` (the default).

## Project workflow

1. Dataset acquisition and class labels.
2. Image resizing, normalization and augmentation.
3. Pretrained EfficientNet-B0 feature extraction and fine-tuning.
4. Validation-based checkpoint selection.
5. Held-out test evaluation and classification report.
6. Streamlit inference, top-3 scores and SQLite history.

## Academic honesty and safety

Do not claim a particular accuracy until you have run `train.py` and evaluated the held-out test split. Report your actual accuracy, precision, recall, F1-score and class-wise errors. This is a research/educational screening tool, not a certified plant pathology diagnosis. Treatment guidance is intentionally general; confirm disease and chemical choice with a local KVK/agriculture officer and follow product labels and local regulations.

## Suggested future upgrades

- Add Indian field-image datasets and local crop varieties.
- Add expert-verified disease severity labels and lesion segmentation.
- Add offline Android deployment (TorchScript/ONNX).
- Add weather/region-aware risk information from trusted sources.
- Add farmer feedback and expert review workflow.
- Add Grad-CAM visual explanations and field-photo robustness evaluation.
