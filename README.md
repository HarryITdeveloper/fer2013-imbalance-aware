# FER-2013 Imbalance-aware Facial Expression Recognition (Topic 4)

This project performs **facial expression recognition** on **FER-2013** (7 classes: angry, disgust, fear, happy, sad, surprise, neutral). 
We **keep a lightweight CNN baseline** (no large backbone) and focus on **class-imbalance-aware training** to improve performance on under-represented classes.

**Final Recommended Model**
- **E5 (Final):** Softmax head + Cross-Entropy (CE) + **Balanced Sampler**

**Baseline Model**
- **E0_match (Baseline):** Softmax head + CE

We report **macro-F1** as the primary metric (more informative under class imbalance), plus per-class F1 and confusion matrices.

---

## 1. Requirements (Software)

- **OS**: Windows / Linux (Tested on Windows 11)
- **Python**: 3.10+ recommended
- **PyTorch**: 2.x + CUDA (GPU recommended)
- **Other dependencies**: See `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: Please install **PyTorch + CUDA** according to your local environment (refer to [pytorch.org](https://pytorch.org/)).

---

## 2. Dataset: FER-2013 (Download & Layout)

The dataset is **not** included in this repository. Please download it from Kaggle:

- **Dataset Identifier**: `astraszab/facial-expression-dataset-image-folders-fer2013`
- **Kaggle Link**: [FER-2013 Image Folders](https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013)

### Step 1: Download
You can download it manually or via the Kaggle API:
```bash
kaggle datasets download -d astraszab/facial-expression-dataset-image-folders-fer2013 --unzip
```

### Step 2: Placement (IMPORTANT)
After unzipping the downloaded Kaggle zip file, you will typically get a folder named archive/.
**Rename or move `archive/` to `data/` in the project root.** 

Your project structure **must** look like this:
```text
<repo_root>/
  ├── data/
  │   ├── train/
  │   │   ├── angry/ ...
  │   │   └── ...
  │   ├── val/
  │   │   └── ...
  │   └── test/
  │       └── ...
  ├── src/
  ├── pretrained/
  └── ...
```

---

## 3. Pretrained Models (for Reproducibility)

We provide two pretrained checkpoints to reproduce our reported numbers:

| Model | Test Acc | Macro-F1 | Disgust F1 | Fear F1 |
| :--- | :--- | :--- | :--- | :--- |
| **E5 (Balanced)** | **0.617** | **0.576** | **0.495** | 0.336 |
| **E0 (Baseline)** | 0.633 | 0.534 | 0.069 | **0.375** |

### How to use:
1. Download the `.pt` files from the [GitHub Releases](https://github.com/HarryITdeveloper/fer2013-imbalance-aware) page.
2. Place them into the `pretrained/` folder:
   - `<repo_root>/pretrained/E5_balanced_best.pt`
   - `<repo_root>/pretrained/E0_match_best.pt`

---

## 4. Evaluation-only (Verified Reproducibility)

Run these commands to verify the metrics. By setting `--epochs 0` and `--resume`, the script will skip training and perform a full evaluation on the test set, generating metrics and confusion matrices.

### Final Model (E5) Evaluation
```bash
python -m src.train --data_root data --exp E5_eval --head softmax --loss ce --sampler balanced --epochs 0 --batch_size 256 --lr 1e-3 --amp --resume pretrained/E5_balanced_best.pt
```

### Baseline (E0) Evaluation
```bash
python -m src.train --data_root data --exp E0_eval --head softmax --loss ce --epochs 0 --batch_size 256 --lr 1e-3 --amp --resume pretrained/E0_match_best.pt
```

### Outputs
After running, check the `runs/<exp>/` folder for:
- `metrics.json`: Detailed F1-scores and accuracy.
- `confusion_matrix.png`: Visual representation of model performance.

---

## 5. (Optional) Training from Scratch

To train the final recommended method (E5) for 30 epochs:
```bash
python -m src.train --data_root data --exp E5_softmax_balanced_30ep --head softmax --loss ce --sampler balanced --epochs 30 --batch_size 256 --lr 1e-3 --amp
```

---

## 6. Repository Structure

- `src/`: Training, evaluation, and data loading logic.
- `tools/`: Helper scripts for generating summary tables/figures.
- `figures/`: Visualizations for the final report.
- `pretrained/`: Directory for storage of `.pt` checkpoints.
- `runs/`: Output directory for experiments (logs, metrics, plots).
- `data/`: Dataset storage (ignored by Git).

---

## 7. Notes / FAQ

**Q: "FileNotFoundError: [Errno 2] No such file or directory: 'data/train'"**  
A: Ensure your dataset folder is renamed to `data` (not `archive`) and is located in the root directory. Check Section 2 for the layout.

**Q: CUDA Out of Memory?**  
A: Reduce the batch size using the flag `--batch_size 128` or `--batch_size 64`.

**Q: How to view training curves?**  
A: We use TensorBoard. Run:
```bash
tensorboard --logdir runs
```
Then open `http://localhost:6006` in your browser.