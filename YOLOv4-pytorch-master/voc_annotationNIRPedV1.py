import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0
# -------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
# -------------------------------------------------------------------#
classes_path = 'model_data/NIRPed_classes.txt'  # 您自己的类别文件

# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹（存放Annotations和ImageSets）
# -------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'

VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

# -------------------------------------------------------#
#   统计目标数量
# -------------------------------------------------------#
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue

        cls_id = classes.index(cls)
        #TODO: Daixb20260722 ==================== 新增跳过逻辑 ====================
        # 移除编号为3的类别（即 'Peo'），不生成标注信息
        # if cls_id == 3:
        #     continue
        #TODO: Daixb20260722 ====================================================
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        list = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in list:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        # 定义图片根目录映射
        image_base = {
            'train': 'E:/Datasets/NIRPed2021/NIRPed/train/images',
            'val': 'E:/Datasets/NIRPed2021/NIRPed/val/images',
            'test': 'E:/Datasets/NIRPed2021/NIRPed/test/images'    }
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                # 读取XML中的filename
                xml_path = os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id))
                tree = ET.parse(xml_path)
                root = tree.getroot()
                filename_elem = root.find('filename')
                if filename_elem is not None:
                    filename = filename_elem.text
                else:
                    filename = image_id + '.jpg'

                # ==================== 修改开始 ====================
                # 根据 image_set 获取首选图片目录
                primary_dir = image_base.get(image_set)
                # 获取备用目录（如果是train则用val，如果是val则用train）
                if image_set == 'train':
                    secondary_dir = image_base.get('val')
                elif image_set == 'val':
                    secondary_dir = image_base.get('train')
                else:
                    secondary_dir = None

                # 先尝试主目录
                candidate_dirs = [primary_dir]
                if secondary_dir is not None:
                    candidate_dirs.append(secondary_dir)

                image_path = None
                for dir_path in candidate_dirs:
                    if dir_path is None:
                        continue
                    test_path = os.path.join(dir_path, filename)
                    if os.path.exists(test_path):
                        image_path = test_path
                        break

                # 如果两个目录都找不到，打印警告并使用主目录（训练时会报错，但会提示）
                if image_path is None:
                    print(f"警告：图片 {filename} 既不在 {primary_dir} 也不在 {secondary_dir}，将使用主目录路径。")
                    image_path = os.path.join(primary_dir, filename)
                # ==================== 修改结束 ====================

                # 写入图片路径
                list_file.write(image_path)
                # 写入标注信息
                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")

        # 统计信息输出
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [classes, str_nums]
        colWidths = [0] * len(tableData)
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")

# D:\env_conda\YoloAll\python.exe D:/YOLO/yolov4-pytorch-master/voc_annotationNIRPedV1.py
# Generate txt in ImageSets.
# train and val size 46863
# train size 42176
# Generate txt in ImageSets done.
# Generate 2007_train.txt and 2007_val.txt for train.
# Generate 2007_train.txt and 2007_val.txt for train done.
# | Ped | 95603 |
# | Bic |  4425 |
# | Mot | 21987 |
# | Peo |  3214 |

# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/voc_annotationNIRPedV1.py
# Generate txt in ImageSets.
# train and val size 46863
# train size 42176
# Generate txt in ImageSets done.
# Generate 2007_train.txt and 2007_val.txt for train.
# Generate 2007_train.txt and 2007_val.txt for train done.
# | Ped | 95603 |
# | Bic |  4425 |
# | Mot | 21987 |
# | Peo |     0 |

# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/voc_annotationNIRPedV1.py
# Generate txt in ImageSets.
# train and val size 47178
# train size 42460
# Generate txt in ImageSets done.
# Generate 2007_train.txt and 2007_val.txt for train.
# Generate 2007_train.txt and 2007_val.txt for train done.
# | Ped | 96025 |
# | Bic |  4439 |
# | Mot | 22162 |
# | Peo |     0 |
#
# Process finished with exit code 0

# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/voc_annotationNIRPedV1.py
# Generate txt in ImageSets.
# train and val size 47178
# train size 42460
# Generate txt in ImageSets done.
# Generate 2007_train.txt and 2007_val.txt for train.
# Generate 2007_train.txt and 2007_val.txt for train done.
# | Ped | 96025 |
# | Bic |  4439 |
# | Mot | 22162 |
# | Peo |  3169 |
#
# Process finished with exit code 0