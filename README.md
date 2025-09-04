# Leukemia Classification with Double Transfer Learning

## Abstract
This project implements an **improvised double transfer learning pipeline** for **Acute Lymphoblastic Leukemia (ALL) classification** using microscopic **Peripheral Blood Smear (PBS) images**.  
We leverage **ResNet-50, EfficientNet-B3, and InceptionV3** backbones with ImageNet pretraining.  
Two-stage fine-tuning is applied:

- **Phase-1**: Binary classification (Normal vs Leukemia) on Dataset-1 (ISBI ALL Challenge).  
- **Phase-2**: Multi-class classification (Benign vs Early Pre-B, Pre-B, Pro-B ALL) on Dataset-2 (Kaggle ALL PBS).

This successive transfer improves performance on small medical datasets by progressively adapting the models to domain-specific features.

---

## Dataset Details

### Dataset-1 (ALL Challenge — TCIA)
- **Images**: 15,135
- **Patients**: 118
- **Classes**: Normal Cells, Leukemia Blasts
- **Source**: The Cancer Imaging Archive (TCIA)  
- **Citation**: Gupta, A. & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019. DOI: [10.7937/tcia.2019.dc64i46r](https://doi.org/10.7937/tcia.2019.dc64i46r)

### Dataset-2 (ALL PBS — Kaggle)
- **Images**: 6,512 (segmented PBS images)
- **Patients**: 89
- **Classes**: Benign (Hematogones), Malignant (Early Pre-B, Pre-B, Pro-B)
- **Source**: Kaggle  
- **Citation**: Mehrad Aria et al. “Acute Lymphoblastic Leukemia (ALL) image dataset.” Kaggle, 2021. DOI: [10.34740/KAGGLE/DSV/2175623](https://doi.org/10.34740/KAGGLE/DSV/2175623)

---

## Methodology

### Data Augmentation
- Random rotations, flips, crops, and zooms
- Color jitter (brightness/contrast/saturation)
- HSV segmentation preprocessing for PBS images

### Double Transfer Learning
1. **Primary Transfer Phase**  
   Train ResNet50, EfficientNetB3, InceptionV3 (ImageNet → Dataset-1).  
   Focus: learn leukemia vs normal morphology.

2. **Secondary Transfer Phase**  
   Fine-tune the same backbones on Dataset-2 (specialized PBS ALL subtypes).  
   Added **custom CNN layers** for PBS-specific features.

---

## Implementation
- Framework: **PyTorch**
- Pretrained Models: ResNet-50, EfficientNet-B3, InceptionV3
- Optimizer: AdamW / Adamax
- Loss: CrossEntropy
- Scheduler: CosineAnnealingLR
- Mixed Precision: Enabled (AMP)

---

## Results (Reported)
| Model          | Phase-2 Accuracy | Eval Acc | Notes |
|----------------|------------------|----------|-------|
| EfficientNetB3 | **95.98%**       | 90.23%   | Best performing |
| ResNet-50      | 83.76%           | 95.50%   | Strong generalization |
| Inception-V3   | 44.1% (10 epochs)| 86.25%   | Underfit with short training |

---

## Novel Extension (Future Work)
- Integrating **Fuzzy Inference System (Mamdani)** with features from double transfer learning models.  
- Aim: Improve robustness on unseen data with fuzzy clustering.

---

## How to Run (This Repo)
### Phase-1 Training
```bash
python -m src.train_phase1 --train_dir /path/to/ds1/train --val_dir /path/to/ds1/val --arch resnet50 --epochs 10 --out_dir checkpoints
```
### Phase-2 Fine-Tuning
```bash
python -m src.finetune_phase2 --train_dir /path/to/ds2/train --val_dir /path/to/ds2/val --phase1_ckpt checkpoints/phase1_resnet50_best.pt --epochs 10 --out_dir checkpoints
```
### Evaluation
```bash
python -m src.evaluate --test_dir /path/to/ds2/test --ckpt checkpoints/phase2_resnet50_best.pt
```
### Predict Single Image
```bash
python -m src.predict_one --image path/to/sample.jpg --ckpt checkpoints/phase2_resnet50_best.pt
```

---

## References
- Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set]. TCIA. DOI: 10.7937/tcia.2019.dc64i46r
- Mehrad Aria et al. (2021). Acute Lymphoblastic Leukemia (ALL) image dataset. Kaggle. DOI: 10.34740/KAGGLE/DSV/2175623
- Ghaderzadeh, M., Aria, M., Hosseini, A, Asadi, F, Bashash, D, Abolghasemi, H. (2022). A fast and efficient CNN model for B-ALL diagnosis. *Int J Intell Syst*, 37: 5113–5133. DOI: 10.1002/int.22753
