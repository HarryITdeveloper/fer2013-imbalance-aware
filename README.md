# FER-2013 Imbalance-aware Facial Expression Recognition (Topic 4)

本项目在 FER-2013（7 类：angry, disgust, fear, happy, sad, surprise, neutral）上进行 facial expression recognition。  
我们固定轻量 CNN baseline，不堆大模型，重点研究 class imbalance 下的训练策略。最终推荐方案为：

- E5 (Final): balanced sampler + cross-entropy (CE)
- 对照基线：E0_match (Baseline): softmax + CE

核心指标以 macro-F1 为主（更能反映类别不平衡），并提供 per-class F1 与 confusion matrix 解释结果差异。

---

## 1. Requirements: Software

- OS: Windows / Linux (tested on Windows 11)
- Python: 3.10+ (recommended)
- PyTorch: 2.x + CUDA (GPU recommended)
- 其他依赖：见 requirements.txt

安装依赖：
```bash
pip install -r requirements.txt
```
建议根据本机 CUDA 版本安装合适的 torch/torchvision（其余依赖可直接用 requirements.txt 安装）。

---

## 2. Dataset: FER-2013 (Download & Prepare)

本仓库不包含数据集（data/ 已在 .gitignore 中排除）。请将 FER-2013 下载并解压到项目根目录下的 data/。

### Option A (Recommended): Kaggle API

1. 安装 Kaggle：

```bash
pip install kaggle
```

2. 配置 Kaggle API Token  
在 Kaggle 个人页面创建 API Token，得到 kaggle.json，放到：

- Windows: C:\Users\<username>\.kaggle\kaggle.json
- Linux/macOS: ~/.kaggle/kaggle.json

3. 下载并解压：

```bash
kaggle datasets download -d astraszab/facial-expression-dataset-image-folders-fer2013 -p data --unzip
```

### Option B: Manual Download

在 Kaggle 数据集页面下载 zip 后，手动解压到 data/。

### Expected folder structure

请确保数据最终目录结构满足如下（示例）：

```
data/
  fer2013/
    train/
      angry/ ...
    val/
      angry/ ...
    test/
      angry/ ...
```

如果你的数据集解压出来目录名不同，请根据实际路径调整 --data_root，或参考 tools/prepare_fer2013.py 的说明。  
--data_root 应指向“包含 train/、val/、test/ 的那一层目录”。常见三种情况如下：

1) 解压后是 data/fer2013/train/ ...  
   -> 使用：--data_root data/fer2013
2) 解压后直接是 data/train/ ...  
   -> 使用：--data_root data
3) 解压后多了一层目录名，例如 data/FER2013_ImageFolders/train/ ...  
   -> 使用：--data_root data/FER2013_ImageFolders

---

## 3. Pretrained models (Reproducibility)

我们提供两份预训练权重用于复现实验数字（建议从 GitHub Releases 下载）：

- pretrained/E5_balanced_best.pt (Final)  
  Test: acc 0.617, macro-F1 0.576, disgust 0.495, fear 0.336
- pretrained/E0_match_best.pt (Baseline)  
  Test: acc 0.633, macro-F1 0.534, disgust 0.069, fear 0.375

下载方式：

- GitHub -> Releases -> v1.0 -> 下载两个 .pt 文件
- 放入项目根目录：pretrained/
  必须放入 pretrained/ 且文件名保持不变：
```
pretrained/
  E5_balanced_best.pt
  E0_match_best.pt
```

---

## 4. Preparation for testing (Evaluation-only, verified)

我们已完成复现性验收：使用 epochs=0 + resume 可稳定复现报告中的指标，并生成输出文件。

### Recommended model (E5) - evaluation only

```bash
python -m src.train --data_root data --exp E5_eval --head softmax --loss ce --sampler balanced --epochs 0 --batch_size 256 --lr 1e-3 --amp --resume pretrained/E5_balanced_best.pt
```

### Baseline model (E0) - evaluation only

```bash
python -m src.train --data_root data --exp E0_eval --head softmax --loss ce --epochs 0 --batch_size 256 --lr 1e-3 --amp --resume pretrained/E0_match_best.pt
```

### Outputs

运行结束后在以下路径生成：

- runs/<exp>/metrics.json
- runs/<exp>/confusion_matrix.png
- (optional) TensorBoard logs: runs/<exp>/events.out.tfevents.*

复现性验收说明：我们已验证 eval-only（epochs=0 + resume）可以复现报告数字。  
结果输出位于 runs/<exp>/metrics.json 与 runs/<exp>/confusion_matrix.png。  
参考指标（Test）：E5_eval acc 0.617 / macro-F1 0.576；E0_eval acc 0.633 / macro-F1 0.534。

---

## 5. (Optional) Training from scratch (30 epochs)

如需从头训练复现最终方案（更耗时）：

```bash
python -m src.train --data_root data --exp E5_softmax_balanced_30ep --head softmax --loss ce --sampler balanced --epochs 30 --batch_size 256 --lr 1e-3 --amp
```

---

## 6. Repo structure

- src/ training & evaluation code
- tools/ helper scripts (summary/figures)
- pretrained/ pretrained checkpoints (download from Releases)
- runs/ experiment outputs (not tracked by git)
- data/ dataset folder (not tracked by git)

---

## 7. Notes / FAQ

- 如果报找不到数据：请检查 --data_root 是否正确，以及 data/ 下的目录结构是否符合预期。
- 如果显存不足：降低 --batch_size（如 128/64）。
- 若想看训练曲线：使用 TensorBoard 打开 runs/。

### TensorBoard (optional)

```bash
tensorboard --logdir runs
```

浏览器打开显示的本地地址即可查看 train/val 曲线。
