# MSGCA-YOLO: Nighttime Pedestrian Detection Model

This repository contains the official implementation for the paper: **"MSGCA-YOLO: A Nighttime Pedestrian Detection Model Combining Multi-scale Global Attention and Coordinate Attention"**.

## 📖 Introduction

Nighttime pedestrian detection is critical for autonomous driving. However, low-contrast imaging exacerbates drastic scale variations and background interference, making reliable detection challenging. We propose **MSGCA-YOLO**, a YOLOv4-based detector that innovatively integrates **Multi-Scale Global Attention (MSGA)** and a **Coordinate-Attention Convolutional Block (CACBL)**:

- **MSGA** utilizes three convolution branches with different receptive fields (3×3, 5×5, and 7×7) to capture multi-scale contextual cues. It further establishes long-range dependencies via a spatial self-attention mechanism, strengthening global perception for targets of varying sizes.
- **CACBL** adaptively re-weights deep semantic features and shallow spatial details through coordinate attention and content-guided strategies, alleviating semantic conflicts during multi-level feature aggregation.

Systematic experiments were conducted on the large-scale nighttime pedestrian dataset **NIRPed**. Results demonstrate that MSGCA-YOLO achieves **90.86% mAP@0.5**, significantly outperforming the baseline. Furthermore, as a plug-and-play component, our modules also yield consistent performance improvements on modern architectures such as YOLOv8s and YOLOv10s.

## 📂 Dataset

The experiments use the third-party public dataset **NIRPed**:

- **Full Dataset (approx. 150GB)**: [Click here to access the download link (Science Data Bank)](https://doi.org/10.57760/sciencedb.010h2)
- **miniNIRPed (Lightweight version)**: Available in the [XiaobiaoDai/NIRPed](https://github.com/XiaobiaoDai/NIRPed) repository.

> **Note**: The authors are not authorized to distribute the NIRPed dataset directly. Please download it using the links above and organize the data in YOLO format.

## 🚀 Environment Setup

- **Python** >= 3.8
- **PyTorch** >= 1.8
- **Ultralytics** (for YOLOv8s/v10s experiments)

```bash
# Install basic dependencies
pip install -r requirements.txt
💻 Training and Testing
This project provides the complete model configuration files and training scripts. Please refer to trainYolo.py and the .yaml configuration files in the models/ directory.

Training
Use the following command to start training the MSGCA-YOLO model (taking the YOLOv10s variant as an example):

bash
python trainYolo.py --data data/nirped.yaml --cfg models/yolov10s_MSGCA.yaml --weights '' --batch-size 56 --epochs 120
Testing / Validation
After training, use the following command for validation or testing:

bash
python val.py --data data/nirped.yaml --weights runs/train/exp/weights/best.pt --batch-size 1
(Note: Please adjust the commands based on your actual file paths)

📜 Citation
If you use the code or ideas from this project in your research, please cite our paper:

bibtex
@article{dai2026msgca,
  title={MSGCA-YOLO: A Nighttime Pedestrian Detection Model Combining Multi-scale Global Attention and Coordinate Attention},
  author={Dai, Xiaobiao and Chen, Zhigang and Dong, Jianhua and Long, Xichen and Hu, Jinkun and Nie, Zexun},
  journal={PLOS ONE},
  year={2026},
  publisher={Public Library of Science},
  doi={10.1371/journal.pone.0362201}
}
⚖️ License
This project is licensed under the MIT License.
```
