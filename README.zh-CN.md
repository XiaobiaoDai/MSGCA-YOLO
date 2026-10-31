# MSGCA-YOLO: Nighttime Pedestrian Detection Model

本项目是论文 **"MSGCA-YOLO: A Nighttime Pedestrian Detection Model Combining Multi-scale Global Attention and Coordinate Attention"** 的官方代码库。

## 📖 简介

夜间行人检测对自动驾驶至关重要，但低对比度成像加剧了尺度剧烈变化和背景干扰，使检测变得困难。本文提出了一种基于 YOLOv4 的检测器 **MSGCA-YOLO**，创新性地集成了**多尺度全局注意力（MSGA）**和**坐标注意力卷积模块（CACBL）**：

- **MSGA** 使用三个不同感受野的卷积核（3×3, 5×5, 7×7）捕获多尺度上下文线索，并通过空间自注意力机制建立长程依赖，强化对不同尺寸目标的全局感知。
- **CACBL** 通过坐标注意力和内容引导策略自适应地重新加权深层语义特征和浅层空间细节，缓解多层特征聚合过程中的语义冲突。

我们在大规模的夜间行人数据集 **NIRPed** 上进行了系统实验。结果表明，MSGCA-YOLO 达到了 **90.86% mAP@0.5**，显著优于基线模型。同时，该模块作为即插即用组件，在 YOLOv8s 和 YOLOv10s 上也取得了稳定的性能提升。

## 📂 数据集

本实验使用第三方公开数据集 **NIRPed**：

- **完整数据集 (约 150GB)**：[点击此处访问下载链接（科学数据银行）](https://doi.org/10.57760/sciencedb.010h2)
- **miniNIRPed（轻量版）**：可在 [XiaobiaoDai/NIRPed](https://github.com/XiaobiaoDai/NIRPed) 仓库中获取。

> **注意**：作者无权直接分发 NIRPed 数据集。请使用上述链接自行下载并按照 YOLO 格式组织数据。

## 🚀 环境配置

- **Python** >= 3.8
- **PyTorch** >= 1.8
- **Ultralytics** (用于 YOLOv8s/v10s 实验)

```bash
# 安装基础依赖
pip install -r requirements.txt
💻 训练与测试
本项目提供了完整的模型配置文件和训练脚本，请参考 trainYolo.py 及 models/ 目录下的 .yaml 配置文件。

训练
使用以下命令启动 MSGCA-YOLO 模型的训练（以 YOLOv10s 变体为例）：

bash
python trainYolo.py --data data/nirped.yaml --cfg models/yolov10s_MSGCA.yaml --weights '' --batch-size 56 --epochs 120
测试
训练完成后，使用以下命令进行验证或测试：

bash
python val.py --data data/nirped.yaml --weights runs/train/exp/weights/best.pt --batch-size 1
(注意：请根据您实际的文件路径调整命令)

📜 引用
如果您在研究中使用了本项目的代码或思路，请引用我们的论文：

bibtex
@article{dai2026msgca,
  title={MSGCA-YOLO: A Nighttime Pedestrian Detection Model Combining Multi-scale Global Attention and Coordinate Attention},
  author={Dai, Xiaobiao and Chen, Zhigang and Dong, Jianhua and Long, Xichen and Hu, Jinkun and Nie, Zexun},
  journal={PLOS ONE},
  year={2026},
  publisher={Public Library of Science},
  doi={10.1371/journal.pone.0362201}
}
⚖️ 许可证
本项目采用 MIT License 许可协议。
```
