# -------------------------------------------------------------------------------------------------------#
#   kmeans虽然会对数据集中的框进行聚类，但是很多数据集由于框的大小相近，聚类出来的9个框相差不大，
#   这样的框反而不利于模型的训练。因为不同的特征层适合不同大小的先验框，shape越小的特征层适合越大的先验框
#   原始网络的先验框已经按大中小比例分配好了，不进行聚类也会有非常好的效果。
# -------------------------------------------------------------------------------------------------------#
import glob
import xml.etree.ElementTree as ET

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")  # 非交互式后端，无 GUI 依赖


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    # -------------------------------------------------------------#
    #   取出一共有多少框
    # -------------------------------------------------------------#
    row = box.shape[0]

    # -------------------------------------------------------------#
    #   每个框各个点的位置
    # -------------------------------------------------------------#
    distance = np.empty((row, k))

    # -------------------------------------------------------------#
    #   最后的聚类位置
    # -------------------------------------------------------------#
    last_clu = np.zeros((row,))

    np.random.seed()

    # -------------------------------------------------------------#
    #   随机选5个当聚类中心
    # -------------------------------------------------------------#
    cluster = box[np.random.choice(row, k, replace=False)]

    iter = 0
    while True:
        # -------------------------------------------------------------#
        #   计算当前框和先验框的宽高比例
        # -------------------------------------------------------------#
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # -------------------------------------------------------------#
        #   取出最小点
        # -------------------------------------------------------------#
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        # -------------------------------------------------------------#
        #   求每一个类的中位点
        # -------------------------------------------------------------#
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near
        if iter % 5 == 0:
            print(f"iter: {iter:d}. avg_iou:{avg_iou(box, cluster):.2f}")
        iter += 1

    return cluster, near


def load_data(path):
    data = []
    # -------------------------------------------------------------#
    #   对于每一个xml都寻找box
    # -------------------------------------------------------------#
    for xml_file in tqdm(glob.glob(f"{path}/*xml")):
        tree = ET.parse(xml_file)
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        if height <= 0 or width <= 0:
            continue

        # -------------------------------------------------------------#
        #   对于每一个目标都获得它的宽高
        # -------------------------------------------------------------#
        for obj in tree.iter("object"):
            # TODO： Daixb202607231050  新增： ========== 新增过滤逻辑 ==========
            name = obj.findtext("name")  # 获取目标名称
            if name == "Peo":  # 如果名称为 "Peo"，跳过
                continue
            # TODO： Daixb202607231050  =================================

            xmin = int(float(obj.findtext("bndbox/xmin"))) / width
            ymin = int(float(obj.findtext("bndbox/ymin"))) / height
            xmax = int(float(obj.findtext("bndbox/xmax"))) / width
            ymax = int(float(obj.findtext("bndbox/ymax"))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


if __name__ == "__main__":
    np.random.seed(0)
    # -------------------------------------------------------------#
    #   运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    #   会生成yolo_anchors.txt
    # -------------------------------------------------------------#
    input_shape = [416, 416]
    anchors_num = 9
    # -------------------------------------------------------------#
    #   载入数据集，可以使用VOC的xml
    # -------------------------------------------------------------#
    path = "VOCdevkit/VOC2007/Annotations"

    # -------------------------------------------------------------#
    #   载入所有的xml
    #   存储格式为转化为比例后的width,height
    # -------------------------------------------------------------#
    print("Load xmls.")
    data = load_data(path)
    print("Load xmls done.")

    # -------------------------------------------------------------#
    #   使用k聚类算法
    # -------------------------------------------------------------#
    print("K-means boxes.")
    cluster, near = kmeans(data, anchors_num)
    print("K-means boxes done.")
    data = data * np.array([input_shape[1], input_shape[0]])
    cluster = cluster * np.array([input_shape[1], input_shape[0]])

    # -------------------------------------------------------------#
    #   绘图
    # -------------------------------------------------------------#
    for j in range(anchors_num):
        plt.scatter(data[near == j][:, 0], data[near == j][:, 1])
        plt.scatter(cluster[j][0], cluster[j][1], marker="x", c="black")
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print("Save kmeans_for_anchors.jpg in root dir.")

    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print(f"avg_ratio:{avg_iou(data, cluster):.2f}")
    print(cluster)

    f = open("yolo_anchors.txt", "w")
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()


# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/kmeans_for_anchors.py
# Load xmls.
# 100%|██████████| 52071/52071 [00:11<00:00, 4609.37it/s]
# Load xmls done.
# K-means boxes.
# iter: 0. avg_iou:0.78
# iter: 5. avg_iou:0.79
# iter: 10. avg_iou:0.79
# iter: 15. avg_iou:0.79
# iter: 20. avg_iou:0.79
# iter: 25. avg_iou:0.79
# K-means boxes done.
# D:\YOLO\YOLOv4-pytorch-master\kmeans_for_anchors.py:149: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()
# Save kmeans_for_anchors.jpg in root dir.
# avg_ratio:0.79
# [[  9.75        34.08888889]
#  [ 11.375       50.84444444]
#  [ 15.6         39.86666667]
#  [ 14.3         62.97777778]
#  [ 18.525       78.57777778]
#  [ 33.15        49.11111111]
#  [ 24.05       106.31111111]
#  [ 35.425      152.53333333]
#  [ 57.85       245.55555556]]
#
# Process finished with exit code 0


# D:\env_conda\YoloAll\python.exe D:/YOLO/YOLOv4-pytorch-master/kmeans_for_anchorsNIRPed.py
# Load xmls.
# 100%|██████████| 52071/52071 [00:11<00:00, 4706.44it/s]
# Load xmls done.
# K-means boxes.
# iter: 0. avg_iou:0.77
# iter: 5. avg_iou:0.79
# iter: 10. avg_iou:0.79
# iter: 15. avg_iou:0.79
# iter: 20. avg_iou:0.79
# iter: 25. avg_iou:0.79
# iter: 30. avg_iou:0.79
# K-means boxes done.
# D:\YOLO\YOLOv4-pytorch-master\kmeans_for_anchorsNIRPed.py:155: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()
# Save kmeans_for_anchors.jpg in root dir.
# avg_ratio:0.79
# [[  9.75        32.35555556]
#  [ 11.7         45.64444444]
#  [ 13.          59.51111111]
#  [ 16.25        68.17777778]
#  [ 27.3         43.33333333]
#  [ 19.825       88.4       ]
#  [ 26.65       112.66666667]
#  [ 37.05       165.24444444]
#  [ 61.425      254.22222222]]
#
# Process finished with exit code 0
