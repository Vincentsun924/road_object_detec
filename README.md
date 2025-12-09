# Road Object Detection with YOLO & BDD100k

This project implements an object detection system for autonomous driving scenarios using the BDD100k dataset and YOLOv11.

## Step-by-Step Guide

### 1. Dataset Preparation (Local)
Before training, we need to convert the BDD100k labels (JSON) into the YOLO format (TXT).

1.  **Download Data:** Download the "100k images" and "labels" folders from the [official site](https://bdd-data.berkeley.edu/).
2.  **Organize and rename Folders:**
    ```text
    data/
    ├── images/
    │   ├──train/
    │   ├──val/
    │   │──test/
    ├── bdd_labels/ (rename and put the "labels" folder here)
    │   ├──train/
    │   ├──val/
    │   │──test/
    ```
3.  **Run Conversion Script:**
    Run the `bdd_to_yolo.py` script to generate the `.txt` labels.
    ```bash
    python bdd_to_yolo.py
    ```
4.  **Drop bdd100k.yaml into `data`:**
    Links the data folders to the YOLO model
5.  **Zip for Upload:**
    Compress the `data` folder into `data` to make it easier to upload to Google Drive.

### 2. Google Colab Setup
Since the dataset is large, we use Google Colab Pro (A100 GPU) (available for free for students) for training.

1.  **Upload Data:** Upload `bdd100k.zip` to your Google Drive MyDrive.
2.  **Open Notebook:** Download `yolo-training-testing.ipynb` and open in Google Colab.
3.  **Change Runtime:** A100 GPU advised (Requires Google Colab Pro)
4.  **Run cells [1] - [5]:**
    * Cell [1]: Installs the Ultralytics dependency, where YOLO resides
    * Cell [2]: Imports the required libraries (YOLO, NumPy)
    * Cell [3]: Mounts Google Drive; required for accessing `data.zip` from Google Colab
    * Cell [4]: Optional. Mainly for debugging
    * Cell [5]: Unzips `data.zip` into the runtime

### 3. Training the Model
We used a combination of data augmentations and model sizes. Estimated training time for 50 epochs small model is 4 hours. Estimated training time for 50 epochs large model is 7 hours.

1.  **Configure `bdd100k.yaml`:** Ensure the path in the YAML file point to `/content/dataset/data`. Train, test, and val should point to `images/train`, `images/test`, and `images/val` respectively.
2.  **Run Training:**
    ```python
    from ultralytics import YOLO
    
    model = YOLO('yolo11m.pt')
    
    results = model.train(
        data='bdd100k.yaml',
        epochs=100,
        imgsz=640,
        batch=32,        # Adjust based on GPU memory
        device='0',
        patience=15,     # Early stopping
        cache=True,      # Cache data in RAM for speed
        
        # Augmentations for Robustness
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.4
    )
    ```

### 4. Evaluation & Testing
1.  **Validate:** Run validation on the test set to get mAP scores.
    ```python
    metrics = model.val(split='test', conf=0.001)
    print(f"mAP50-95: {metrics.box.map}")
    ```
2.  **Inference:** Run the model on a sample video or image.
    ```python
    model.predict("test_video.mp4", save=True)
    ```

### 5. Saving Results
Don't lose your trained model!

1.  **Copy to Drive:**
    ```python
    !cp /content/runs/detect/train/weights/best.pt /content/drive/MyDrive# filepath: /Users/beaulin/DS 340 Project/Analysis/road_object_detec/README.md
# Road Object Detection with YOLO & BDD100k

This project implements an object detection system for autonomous driving scenarios using the BDD100k dataset and YOLOv11.

## 🚀 Step-by-Step Guide

### 1. Dataset Preparation (Local)
Before training, we need to convert the BDD100k labels (JSON) into the YOLO format (TXT).

1.  **Download Data:** Download the BDD100k images and labels from the [official site](https://bdd-data.berkeley.edu/).
2.  **Organize Folders:**
    ```text
    bdd10k/
    ├── images/
    │   ├── 10k/ (train/val/test)
    ├── labels/ (will be generated)
    ├── bdd_labels/ (put original .json files here)
    ```
3.  **Run Conversion Script:**
    Run the `bdd_to_yolo.py` script to generate the `.txt` labels.
    ```bash
    python bdd_to_yolo.py
    ```
4.  **Zip for Upload:**
    Compress the `bdd10k` folder into `bdd100k.zip` to make it easier to upload to Google Drive.

### 2. Google Colab Setup
Since the dataset is large, we use Google Colab Pro (A100/V100 GPU) for training.

1.  **Upload Data:** Upload `bdd100k.zip` to your Google Drive.
2.  **Open Notebook:** Open `yolo-training-testing.ipynb` in Google Colab.
3.  **Mount Drive & Unzip:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    
    !cp "/content/drive/MyDrive/bdd100k.zip" "/content/bdd100k.zip"
    !unzip -q "/content/bdd100k.zip" -d "/content/dataset"
    ```
4.  **Install Dependencies:**
    ```python
    !pip install ultralytics
    ```

### 3. Training and Evaluating the Model
We use YOLOv11m (Medium) for a balance of speed and accuracy.

1.  **Configure `bdd100k.yaml`:** Ensure the paths in the YAML file point to `/content/dataset/bdd10k`.
2.  **Run Training:** Each subsequent cell indicates a separate training attempt. Hyperparameters have been tuned for speed, while trying to maintain accuracy.
    * `bdd_colab_run` - (n) test run w/ 15 epochs
    * `bdd_colab_run2` - (s) default data augmentations w/ 50 epochs
    * `bdd_colab_run3` - (s) added erasure, copy-paste data augmentations w/ 50 epochs
    * `bdd_colab_run4` - (m) added erasure, copy-paste data augmentations w/ 50 epochs

### 5. Saving Results
Don't lose your trained model! Output directory name can be changed at will

1.  **Copy to Drive:**
    ```
    !cp -r runs/detect/[train_run_name] '/content/drive/MyDrive/DS 340 Project/YOLO_Results/'
    !cp -r runs/detect/[val_run_name] '/content/drive/MyDrive/DS 340 Project/YOLO_Results/'
    ```
2.  **Download:**
    ```
    !zip -r /content/[train_run_name].zip /content/runs/detect/[train_run_name]
    !zip -r /content/[val_run_name].zip /content/runs/detect/[val_run_name]
    from google.colab import files
    files.download('/content/[test_run_name].zip')
    files.download('/content/[val_run_name].zip')
    ```

### 6. Predicting Detection Boxes
    You can use the weights from previous training runs in order to detect objects on a custom image. The previous scripts save the weights with the best accuracy (`best.pt`), the worst accuracy (`worst.pt`), and every 5 epochs (`epoch[#].pt`).
    from ultralytics import YOLO
    ```
    model = YOLO('[path_to_weight]')

    results = model.predict(
        source='[path_to_image]', 
        save=True, 
        show_labels=False,
        show_conf=False,
        conf=0.25,
        iou=0.45
    )
    ```