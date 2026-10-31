import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器/脚本环境
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO  # ultralytics  __version__ = "8.4.87"

if __name__ == '__main__':
    # model = YOLO('./ultralytics/cfg/models/v8/yolov8s-MSGCA.yaml')    # 加载自定义模型结构（带 CA）
    # model.load('../YOLOvAll0/weightsOk/yolov8s.pt')  # 加载预训练权重（只匹配的层会被加载，新加的 CA 层随机初始化）；也可使用绝对路径
    model = YOLO('./ultralytics/cfg/models/v10/yolov10s-MSGCA.yaml')    # 加载自定义模型结构（带 CA）
    model.load('../YOLOvAll0/weightsOk/yolov10s.pt')  # 加载预训练权重（只匹配的层会被加载，新加的 CA 层随机初始化）；也可使用绝对路径

    # 开始训练
    results = model.train(
        # data='E:/Datasets/NIRPed2021/NIRPed/NIRPedTrain.yaml',  #E:/Datasets/NIRPed2021/NIRPed  —— /tmp1/Datasets/NIRPed
        data='E:/Datasets/NIRPed2021/miniNIRPed/miniNIRPedTrain.yaml',
        epochs=32,
        imgsz=640,
        batch=32,                 # --- 批次大小 (Batch Size) ---单卡batch=32，3卡总计96[reference:1]
        device=0,                # 多卡核心设置[0, 1, 2] --- 指定3张GPU的ID
        workers=8,              #--- 工作线程数 (Workers) --- 建议为 8 * GPU数量[reference:2]
        lr0=0.01,                 #--- 学习率 (Learning Rate) --- 初始学习率可保持不变
        patience=10,            # 早停耐心值
        project='./runs/train',
        # name='yolov10s_NIRPed4cls_IgnTrain-IgnVal',   #NoIgnTrain NoIgnVal
        name='yolov10s_miniNIRPed4cls_IgnTrain-IgnVal0001',   #NoIgnTrain NoIgnVal
        exist_ok=True,
        pretrained=False,           # 已经手动加载，避免自动下载覆盖
        seed=42,                       # 固定随机种子便于复现
        deterministic=False,       # 关闭确定性
    )
#     # --- 其他推荐设置 ---
#     amp=True,                   # 启用混合精度训练 (Automatic Mixed Precision)
#     sync_bn=True,               # 同步批归一化 (SyncBatchNorm)
#     close_mosaic=10,            # 最后10个epoch关闭马赛克增强
# TODO：yolo26n  yolo26s  yolo26l  yolo26m  yolo26x
# TODO：yolo11n   yolo11s   yolo11l   yolo11m   yolo11x
# TODO：yolov10n   yolov10s   yolov10l   yolov10m   yolov10x
# TODO：yolov8n   yolov8s   yolov8l   yolov8m   yolov8x
# TODO：yolov5nu   yolov5su   yolov5lu   yolov5mu   yolov5xu