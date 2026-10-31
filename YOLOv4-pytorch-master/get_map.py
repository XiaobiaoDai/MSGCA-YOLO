import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yoloValidation import  YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 4     #TODO： Daixb202607212240   改进：0  -  4
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可 D:\YOLO\YOLOv4-pytorch-master\logs\loss_2026_07_24_21_24_57
    #--------------------------------------------------------------------------------------#
    # model_path      = 'model_data/ep110-loss0.036-val_loss0.026_3cls.pth'     #TODO： Daixb202607212240   新增：ep110-loss0.036-val_loss0.026_3cls
    model_path      = 'logs/loss_2026_07_25_23_3clsAnchor_mAP90.6/best_epoch_weights.pth'     #TODO： Daixb202607212240   新增：ep110-loss0.036-val_loss0.026_3cls
    # model_path      = 'logs/loss_2026_07_25_13_3clsAnchor_mAP89.1/best_epoch_weights.pth'     #TODO： Daixb202607212240   新增：ep110-loss0.036-val_loss0.026_3cls
    # model_path      = 'logs/loss_2026_07_24_21_24_57/ep120-loss0.054-val_loss0.043_3cls.pth'     #TODO： Daixb202607212240   新增：ep110-loss0.036-val_loss0.026_3cls
    # model_path      = 'model_data/ep010-loss0.099-val_loss0.067_3clsAnchor.pth'     #TODO： Daixb202607212240   新增：ep110-loss0.036-val_loss0.026_3cls
    classes_path    = 'model_data/NIRPed3cls.txt'  #TODO： Daixb202607212240   改进：voc_classes.txt  —  NIRPed_classes.txt
    #--------------------------------------------------------------------------------------#
    #   anchors_path    代表先验框对应的txt文件，一般不修改。
    #   anchors_mask    用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    # anchors_path    = 'model_data/yolo_anchorsVOC.txt'   #TODO: Daixb20260721 改进：yolo_anchors  — yolo_anchorsVOC  —  yolo_anchorsNIRPed3cls
    anchors_path    = 'model_data/yolo_anchorsNIRPed3cls.txt'   #TODO: Daixb20260721 改进：yolo_anchors  — yolo_anchorsVOC  —  yolo_anchorsNIRPed3cls
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #----------------------------------------------------------------------------------------------------------------------------#

    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.4   #TODO:  Daixb20260725 改进：0.5 — 0.4
    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   
    #   该值一般不调整。
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5    #TODO:  Daixb20260725 改进：0.5 — 0.4
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5  #TODO:  Daixb20260725 改进：0.5 — 0.4
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    imageNIRPedTrain_path  = 'E:/Datasets/NIRPed2021/NIRPed/train/images'  #TODO: Daixb20260721 改进：增加训练集路径
    imageNIRPedVal_path  = 'E:/Datasets/NIRPed2021/NIRPed/val/images'  #TODO: Daixb20260721 改进：增加验证集路径
    # imageNIRPedTrain_path  = '/tmp1/Datasets/NIRPed/train/images'  #TODO: Daixb20260721 改进：增加训练集路径
    # imageNIRPedVal_path  = '/tmp1/Datasets/NIRPed/val/images'  #TODO: Daixb20260721 改进：增加验证集路径
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    map_out_path    = f'map_MINOVERLAP{MINOVERLAP}_{Path(model_path).stem}_{Path(anchors_path).stem}'

    # image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    # image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/val.txt")).read().strip().split()
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/trainval.txt")).read().strip().split()


    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, num_classes = get_classes(classes_path)  # 获取classes
    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        # yolo = YOLO(confidence = confidence, nms_iou = nms_iou)  #TODO: Daixb20260722 原代码。
        yolo = YOLO( model_path=model_path,   classes_path=classes_path,  anchors_path=anchors_path,  confidence=confidence, nms_iou=nms_iou)  #TODO: Daixb20260722 改进

        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image_path  = os.path.join(imageNIRPedVal_path, image_id+".png")
            if  not os.path.exists(image_path): #TODO: Daixb20260722 新增判断
                image_path = os.path.join(imageNIRPedTrain_path, image_id + ".png")

            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))

            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")


# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/get_map.py
# Load model.
# model_data/best_epoch_weights.pth model, anchors, and classes loaded.
# Configurations:
# ----------------------------------------------------------------------
# |                     keys |                                   values|
# ----------------------------------------------------------------------
# |               model_path |        model_data/best_epoch_weights.pth|
# |             classes_path |            model_data/NIRPed_classes.txt|
# |             anchors_path |              model_data/yolo_anchors.txt|
# |             anchors_mask |        [[6, 7, 8], [3, 4, 5], [0, 1, 2]]|
# |              input_shape |                               [416, 416]|
# |               confidence |                                    0.001|
# |                  nms_iou |                                      0.5|
# |          letterbox_image |                                    False|
# |                     cuda |                                     True|
# ----------------------------------------------------------------------
# Load model done.
# Get predict result.
#   0%|          | 0/4687 [00:00<?, ?it/s]D:\YOLO\YOLOv4-pytorch-master\utils\utils_bbox.py:88: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\tensor\python_tensor.cpp:80.)
#   anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
# 100%|██████████| 4687/4687 [03:29<00:00, 22.37it/s]
# Get predict result done.
# Get ground truth result.
# 100%|██████████| 4687/4687 [00:02<00:00, 2152.54it/s]
# Get ground truth result done.
# Get map.
# 60.25% = Bic AP 	||	score_threhold=0.5 : F1=0.60 ; Recall=57.73% ; Precision=61.42%
# Class: Bic
#   Scale small: AP@0.5=0.00%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale medium: AP@0.5=37.57%, MR=45.02%, Recall=54.98%, Precision=39.04%, F1=0.46
#   Scale large: AP@0.5=25.50%, MR=36.77%, Recall=63.23%, Precision=22.37%, F1=0.33
# 73.62% = Mot AP 	||	score_threhold=0.5 : F1=0.70 ; Recall=58.02% ; Precision=87.20%
# Class: Mot
#   Scale small: AP@0.5=0.12%, MR=83.33%, Recall=16.67%, Precision=0.07%, F1=0.00
#   Scale medium: AP@0.5=34.05%, MR=52.62%, Recall=47.38%, Precision=44.99%, F1=0.46
#   Scale large: AP@0.5=52.09%, MR=23.50%, Recall=76.50%, Precision=42.57%, F1=0.55
# 92.43% = Ped AP 	||	score_threhold=0.5 : F1=0.87 ; Recall=84.72% ; Precision=88.78%
# Class: Ped
#   Scale small: AP@0.5=0.04%, MR=92.86%, Recall=7.14%, Precision=0.01%, F1=0.00
#   Scale medium: AP@0.5=44.65%, MR=22.38%, Recall=77.62%, Precision=47.47%, F1=0.59
#   Scale large: AP@0.5=58.68%, MR=5.37%, Recall=94.63%, Precision=42.64%, F1=0.59
# 6.10% = Peo AP 	||	score_threhold=0.5 : F1=0.01 ; Recall=0.68% ; Precision=50.00%
# Class: Peo
#   Scale small: AP@0.5=0.00%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale medium: AP@0.5=3.12%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale large: AP@0.5=4.47%, MR=98.20%, Recall=1.80%, Precision=50.00%, F1=0.03
# mAP = 58.10%
# Get map done.
#
# Process finished with exit code 0

# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/get_map.py
# Load model.
# model_data/best_epoch_weights.pth model, anchors, and classes loaded.
# Configurations:
# ----------------------------------------------------------------------
# |                     keys |                                   values|
# ----------------------------------------------------------------------
# |               model_path |        model_data/best_epoch_weights.pth|
# |             classes_path |            model_data/NIRPed_classes.txt|
# |             anchors_path |              model_data/yolo_anchors.txt|
# |             anchors_mask |        [[6, 7, 8], [3, 4, 5], [0, 1, 2]]|
# |              input_shape |                               [416, 416]|
# |               confidence |                                    0.001|
# |                  nms_iou |                                      0.5|
# |          letterbox_image |                                    False|
# |                     cuda |                                     True|
# ----------------------------------------------------------------------
# Load model done.
# Get predict result.
#   0%|          | 0/4687 [00:00<?, ?it/s]D:\YOLO\YOLOv4-pytorch-master\utils\utils_bbox.py:88: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\tensor\python_tensor.cpp:80.)
#   anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
# 100%|██████████| 4687/4687 [04:22<00:00, 17.88it/s]
#   0%|          | 0/4687 [00:00<?, ?it/s]Get predict result done.
# Get ground truth result.
# 100%|██████████| 4687/4687 [00:34<00:00, 136.58it/s]
# Get ground truth result done.
# Get map.
# 60.25% = Bic AP 	||	score_threhold=0.5 : F1=0.60 ; Recall=57.73% ; Precision=61.42%
# Class: Bic
#   Scale small: AP@0.5=0.00%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale medium: AP@0.5=37.57%, MR=45.02%, Recall=54.98%, Precision=39.04%, F1=0.46
#   Scale large: AP@0.5=25.50%, MR=36.77%, Recall=63.23%, Precision=22.37%, F1=0.33
# 73.62% = Mot AP 	||	score_threhold=0.5 : F1=0.70 ; Recall=58.02% ; Precision=87.20%
# Class: Mot
#   Scale small: AP@0.5=0.12%, MR=83.33%, Recall=16.67%, Precision=0.07%, F1=0.00
#   Scale medium: AP@0.5=34.05%, MR=52.62%, Recall=47.38%, Precision=44.99%, F1=0.46
#   Scale large: AP@0.5=52.09%, MR=23.50%, Recall=76.50%, Precision=42.57%, F1=0.55
# 92.43% = Ped AP 	||	score_threhold=0.5 : F1=0.87 ; Recall=84.72% ; Precision=88.78%
# Class: Ped
#   Scale small: AP@0.5=0.04%, MR=92.86%, Recall=7.14%, Precision=0.01%, F1=0.00
#   Scale medium: AP@0.5=44.65%, MR=22.38%, Recall=77.62%, Precision=47.47%, F1=0.59
#   Scale large: AP@0.5=58.68%, MR=5.37%, Recall=94.63%, Precision=42.64%, F1=0.59
# 6.10% = Peo AP 	||	score_threhold=0.5 : F1=0.01 ; Recall=0.68% ; Precision=50.00%
# Class: Peo
#   Scale small: AP@0.5=0.00%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale medium: AP@0.5=3.12%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale large: AP@0.5=4.47%, MR=98.20%, Recall=1.80%, Precision=50.00%, F1=0.03
# mAP = 58.10%
#
# Overall MR per class (at score_threhold=0.50):
#   Bic: MR = 36.77%
#   Mot: MR = 23.50%
#   Ped: MR = 5.37%
#   Peo: MR = 98.20%
#   Average MR (all classes): 40.96%
#
# Average MR per scale:
#   small: Average MR = 94.05%
#   medium: Average MR = 55.01%
#   large: Average MR = 40.96%
# Get map done.
#
# Process finished with exit code 0



# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/get_map.py
# Load model.
# model_data/best_epoch_weights.pth model, anchors, and classes loaded.
# Configurations:
# ----------------------------------------------------------------------
# |                     keys |                                   values|
# ----------------------------------------------------------------------
# |               model_path |        model_data/best_epoch_weights.pth|
# |             classes_path |            model_data/NIRPed_classes.txt|
# |             anchors_path |              model_data/yolo_anchors.txt|
# |             anchors_mask |        [[6, 7, 8], [3, 4, 5], [0, 1, 2]]|
# |              input_shape |                               [416, 416]|
# |               confidence |                                    0.001|
# |                  nms_iou |                                      0.5|
# |          letterbox_image |                                    False|
# |                     cuda |                                     True|
# ----------------------------------------------------------------------
# Load model done.
# Get predict result.
#   0%|          | 0/4687 [00:00<?, ?it/s]D:\YOLO\YOLOv4-pytorch-master\utils\utils_bbox.py:88: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\tensor\python_tensor.cpp:80.)
#   anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
# 100%|██████████| 4687/4687 [03:39<00:00, 21.32it/s]
# Get predict result done.
# Get ground truth result.
# 100%|██████████| 4687/4687 [00:02<00:00, 2013.27it/s]
# Get ground truth result done.
# Get map.
# 74.38% = Bic AP 	||	score_threhold=0.5 : F1=0.72 ; Recall=69.31% ; Precision=74.77%
# Class: Bic
#   Scale small: AP@0.5=0.00%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale medium: AP@0.5=44.67%, MR=36.66%, Recall=63.34%, Precision=45.81%, F1=0.53
#   Scale large: AP@0.5=34.24%, MR=18.71%, Recall=81.29%, Precision=29.30%, F1=0.43
# 82.73% = Mot AP 	||	score_threhold=0.5 : F1=0.77 ; Recall=67.81% ; Precision=88.73%
# Class: Mot
#   Scale small: AP@0.5=0.12%, MR=83.33%, Recall=16.67%, Precision=0.06%, F1=0.00
#   Scale medium: AP@0.5=43.27%, MR=39.22%, Recall=60.78%, Precision=50.25%, F1=0.55
#   Scale large: AP@0.5=53.93%, MR=19.80%, Recall=80.20%, Precision=38.86%, F1=0.52
# 94.75% = Ped AP 	||	score_threhold=0.5 : F1=0.89 ; Recall=86.55% ; Precision=91.19%
# Class: Ped
#   Scale small: AP@0.5=0.06%, MR=85.71%, Recall=14.29%, Precision=0.02%, F1=0.00
#   Scale medium: AP@0.5=47.59%, MR=19.65%, Recall=80.35%, Precision=49.21%, F1=0.61
#   Scale large: AP@0.5=54.61%, MR=4.77%, Recall=95.23%, Precision=42.97%, F1=0.59
# 8.57% = Peo AP 	||	score_threhold=0.5 : F1=0.03 ; Recall=1.69% ; Precision=41.67%
# Class: Peo
#   Scale small: AP@0.5=0.00%, MR=100.00%, Recall=0.00%, Precision=0.00%, F1=0.00
#   Scale medium: AP@0.5=4.74%, MR=98.37%, Recall=1.63%, Precision=25.00%, F1=0.03
#   Scale large: AP@0.5=4.82%, MR=98.20%, Recall=1.80%, Precision=16.67%, F1=0.03
# mAP = 65.11%
#
# Overall MR per class (at score_threhold=0.50):
#   Bic: MR = 18.71%
#   Mot: MR = 19.80%
#   Ped: MR = 4.77%
#   Peo: MR = 98.20%
#   Average MR (all classes): 35.37%
#
# Average MR per scale:
#   small: Average MR = 92.26%
#   medium: Average MR = 48.47%
#   large: Average MR = 35.37%
# Get map done.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.396
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.649
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.114
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.362
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.452
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.527
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.529
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.525
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.502
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
# Process finished with exit code 0