# import torch
import matplotlib

from ultralytics import YOLO

matplotlib.use("Agg")  # 非交互式后端，无 GUI 依赖

if __name__ == "__main__":
    # 加载模型（如果本地已有权重文件，直接指定路径）
    model = YOLO("D:/YOLO/YOLOvAll0/weightsOk/yolo8s.pt")  # 或者 'path/to/your/model.pt'
    # # 终端(Terminal)执行检测指令：yolo predict model=yolo26l.pt source='./images/Data20210315213831_712196N850F12.png' #TODO：yolo26n  yolo26s  yolo26l  yolo26m  yolo26x
    # #>yolo predict model=yolo11s.pt source='./images/Data20210315213831_712196N850F12.png'  #TODO：yolo11n   yolo11s   yolo11l   yolo11m   yolo11x
    # # yolo predict model=yolov10x.pt source='./images/Data20210315213831_712196N850F12.png'   #TODO：yolov10n   yolov10s   yolov10l   yolov10m   yolov10x
    # #>yolo predict model=yolo8s.pt source='./images/Data20210315213831_712196N850F12.png'  #TODO：yolo8n   yolo8s   yolo8l   yolo8m   yolov8x
    # #>yolo predict model=yolo5s.pt source='./images/Data20210315213831_712196N850F12.png'  #TODO：yolov5n   yolov5su   yolov5l   yolov5m   yolov5xu
    # TODO 验证:
    metrics = model.val(
        data="E:/Datasets/NIRPed2021/miniNIRPed/NIRPed.yaml",
        device=0,  # 使用GPU，如果是CPU则设为 'cpu'
        batch=16,  # 根据显存调整
        imgsz=640,
        workers=4,  # 临时设置为0，避免多进程问题（但会慢一点）workers=4
        split="val",  #  val   test
    )

    # 打印结果
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    print(
        f"MR_AllCls @ IoU=0.5, conf=0.5: {metrics.box.mr_conf50:.4f}"
    )  # TODO：字典，键为类别索引，值为所有类的 MR的均值
    print(
        f"MR @ IoU=0.5, conf=0.5: {metrics.box.mr_conf50_per_class[0]:.4f}"
    )  # TODO：字典，键为0类别索引，值为该类的 MR
    print(f"LAMR50: {metrics.box.lamr[0]:.4f}")  # TODO：字典，键为类别索引，值为0类的 LAMR

    # if metrics.box.lamr:    #TODO： 计算平均 LAMR（通常用于整体评估）
    #     avg_lamr = sum(metrics.box.lamr.values()) / len(metrics.box.lamr)
    #     print(f"Average LAMR: {avg_lamr:.4f}")
