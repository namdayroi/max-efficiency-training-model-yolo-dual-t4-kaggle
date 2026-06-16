# 🗑️ YOLO11 Dataset Configuration & Training Guide on Kaggle

This document focuses on guiding how to organize dataset in YOLO format, steps to customize the notebook for your own project, and deep-diving into training settings in **Cell 5** for the best performance.

---

## 🌟 Connect & Support the Author

If you find this notebook and guide useful, please consider following and supporting my work on Kaggle! Check out my other datasets, models, and notebooks here:

👉 **Kaggle Profile:** [@namproovipp](https://www.kaggle.com/namproovipp)

*Upvotes and follows are greatly appreciated! Thank you!*

---

## 📂 Part 1: Standard Dataset Folder Tree Structure (YOLO Format)

To enable YOLO to recognize and learn from your data, the folder structure must be organized exactly as shown below.

### 1. Input Dataset Structure (Unsplit)
If your uploaded Kaggle dataset puts all images and labels together in flat directories, the standard structure is as follows:
```text
dataset/
├── classes.txt          # File containing the list of class names (one class per line)
├── images/              # Directory containing all images (.jpg, .png, .jpeg)
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── labels/              # Directory containing annotations (.txt) corresponding to each image
    ├── img_001.txt
    ├── img_002.txt
    └── ...
```
> [!IMPORTANT]
> **YOLO Label File Rules (.txt):**
> * The label file name must exactly match the image file name (e.g., `img_001.jpg` must correspond to `img_001.txt`).
> * Each line in the `.txt` file represents one object in the format: 
>   `<class_id> <x_center> <y_center> <width> <height>`
>   *(Where coordinate and dimension values are normalized to range from `0` to `1`)*

### 2. Dataset Structure After Train/Val Split (Automatically created in Cell 4)
After running the dataset splitting script in Cell 4, the system will automatically copy and split the dataset to the writable directory `/kaggle/working/dataset/` under the standard YOLO tree structure:
```text
/kaggle/working/dataset/
├── images/
│   ├── train/           # Images used for training (80%)
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/             # Images used for validation/testing (20%)
│       ├── img_002.jpg
│       └── ...
└── labels/
    ├── train/           # Labels corresponding to the training set
    │   ├── img_001.txt
    │   └── ...
    └── val/             # Labels corresponding to the validation set
        ├── img_002.txt
        └── ...
```

---

## 🛠️ Part 2: What to Modify in the Notebook for a New Project

When applying this notebook to your own new dataset or object detection project, modify the following parts:

### 1. Change the Original Dataset Path
In **Cell 4**, change the `input_dir` path to your new dataset directory on Kaggle:
```python
# Change this path to your dataset path on Kaggle
input_dir = '/kaggle/input/<your-dataset-folder-name>/dataset'
```

### 2. Adjust the Dataset Split Ratio (Train/Val Split)
If you want to increase or decrease the amount of images used for evaluating the model, edit the split ratio at this line in **Cell 4**:
```python
# Default is 80% Train and 20% Val. Change 0.8 to your desired ratio (e.g. 0.9 for 90/10 split)
split_idx = int(len(images) * 0.8) 
```

### 3. Adjust GPU Settings According to Kaggle Accelerator Settings
If you change the hardware accelerator configuration (Accelerator) in your Kaggle Notebook settings:
* Using **Dual T4 GPUs**: Keep the `device='0,1'` argument in Cell 5.
* Using **Single T4 GPU** or **L4 GPU**: Change to `device=0`.
* Using **CPU**: Change to `device='cpu'` (Highly discouraged as training will be extremely slow).

---

## ⚡ Part 3: Deep Dive & Optimal Configuration for Cell 5

**Cell 5** is the core of the training process:
```bash
!yolo detect train model=yolo11n.pt data=/kaggle/working/data_kaggle.yaml epochs=50 imgsz=640 batch=60 device='0,1' workers=2 cache=true
```

Below is a detailed analysis of each parameter and guidelines on how to configure them to achieve the best results for your project.

### 1. `model=yolo11n.pt` (Choosing Model Size)
YOLO11 models come in various sizes, ranging from lightweight to heavy. The choice depends on the balance between **accuracy** and **speed**:
* **`yolo11n.pt` (Nano):** Lightest (~3M parameters). Extremely fast training, consumes very few resources. Best for quick tests or deploying on low-end edge devices (Raspberry Pi, mobile phones).
* **`yolo11s.pt` (Small):** Good balance between speed and accuracy (~9M parameters).
* **`yolo11m.pt` (Medium):** Suitable for most average real-world problems.
* **`yolo11l.pt` (Large) / `yolo11x.pt` (Extra Large):** Highest accuracy but large size, takes a long time to train, and is heavy. Only use when smaller models cannot detect difficult details.

### 2. `epochs=50` (Training Epochs)
* The number of times the entire dataset passes through the neural network. 50 epochs is usually only enough for initial convergence testing.
* **Recommended:** Set `epochs=100` to `epochs=300` for deeper learning.
* **Optimization Tip:** Add the `patience=50` parameter (the model will automatically stop if the accuracy on the Val set does not improve for 50 consecutive epochs). This helps save your Kaggle GPU quota once the model has reached its learning limit.

### 3. `imgsz=640` (Input Image Size)
* Images will be automatically resized to a square `640x640` before being fed into the model.
* **When to decrease (`imgsz=320` or `imgsz=416`):** When the GPU VRAM is too low or you need the model to run at very high FPS during real-time inference.
* **When to increase (`imgsz=1028` or higher):** When the target objects to detect are very small compared to the overall image.

### 4. `batch=60` (Batch Size)
* The number of images loaded into the GPU at once to calculate gradients.
* **Optimization:** Larger batch sizes make training more stable and faster, but require larger VRAM.
* **Troubleshooting:** If you encounter a red screen error saying `CUDA out of memory`, decrease this value to powers of 2 such as `batch=32`, `batch=16`, or `batch=8` until it runs stably.

### 5. `device='0,1'` (Using Multi-GPU on Kaggle)
* Kaggle allows using 2 T4 GPUs in parallel (Dual T4). Setting `'0,1'` will enable Distributed Data Parallel (DDP) training, speeding up the training process by almost 2x.
* If using a single GPU, set to `device=0`.

### 6. `cache=true` (Image Caching)
* Caches pre-processed images into the system RAM instead of reading them continuously from the hard disk. Significantly speeds up training.
* **Note:** Only enable `cache=true` when your dataset size is moderate (under 10GB). If the dataset is too large, caching will fill up the Kaggle VM RAM and crash the system. In that case, modify it to `cache=false`.
