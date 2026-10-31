import  ultralytics
from ultralytics import YOLO

print(ultralytics.__version__)
# # 终端(Terminal)执行检测指令：yolo predict model=yolo26l.pt source='./images/Data20210315213831_712196N850F12.png' #TODO：yolo26n  yolo26s  yolo26l  yolo26m  yolo26x
# #>yolo predict model=yolo11s.pt source='./images/Data20210315213831_712196N850F12.png'  #TODO：yolo11n   yolo11s   yolo11l   yolo11m   yolo11x
# # yolo predict model=yolov10x.pt source='./images/Data20210315213831_712196N850F12.png'   #TODO：yolov10n   yolov10s   yolov10l   yolov10m   yolov10x
# #>yolo predict model=yolo8s.pt source='./images/Data20210315213831_712196N850F12.png'  #TODO：yolo8n   yolo8s   yolo8l   yolo8m   yolo8x
# #>yolo predict model=yolo5s.pt source='./images/Data20210315213831_712196N850F12.png'  #TODO：yolo5n   yolo5s   yolo5l   yolo5m   yolo5x
#
# model = YOLO('yolo8s.pt')# 加载模型 (会自动下载 .pt 文件)  yolov8s  yolov8n  yolov8m  yolov8l  yolo26l
# # results = model('https://ultralytics.com/images/bus.jpg')# 对图片进行预测
# results = model(r"E:\Datasets\NIRPed2021\miniNIRPed\images&pickles\val\Data20210315213831_712196N850F12.png") # 执行预测： 可指定图片/视频路径
# # results = model(r"E:\Datasets\NIRPed2021\miniNIRPed\images&pickles\val\Data20210315213831_178154N850F12.png") # 执行预测： 可指定图片/视频路径
# # results = model(r"E:\Datasets\NIRPed2021\miniNIRPed\images&pickles\val\Data20200525214032_985570N850F12.png") # 执行预测： 可指定图片/视频路径
#
# results[0].show()# 显示结果

model = YOLO("yolo26s.pt")# 加载你的模型
metrics = model.val(data="E:/Datasets/NIRPed2021/miniNIRPed/NIRPed.yaml")# 运行验证

# 打印关键指标
print(f"mAP50-95: {metrics.box.map}")    # 平均精度 (mAP) @ [0.5:0.95]
print(f"mAP50: {metrics.box.map50}")      # mAP @ IoU=0.5
print(f"mAP75: {metrics.box.map75}")      # mAP @ IoU=0.75