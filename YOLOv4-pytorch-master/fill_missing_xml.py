import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from utils.utils import get_classes

# ======================== 配置 ========================
classes_path = 'model_data/NIRPed4cls.txt'
VOCdevkit_path = 'VOCdevkit'
image_base = {
    'train': 'E:/Datasets/NIRPed2021/NIRPed/train/images',
    'val':   'E:/Datasets/NIRPed2021/NIRPed/val/images',
    'test':  'E:/Datasets/NIRPed2021/NIRPed/test/images'
}
label_base = {
    'train': 'E:/Datasets/NIRPed2021/NIRPed/train/labels04',
    'val':   'E:/Datasets/NIRPed2021/NIRPed/val/labels04',
    'test':  'E:/Datasets/NIRPed2021/NIRPed/test/labels04'
}
# =====================================================

classes, _ = get_classes(classes_path)

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            return w, h
    except Exception as e:
        print(f"警告：无法读取图片 {image_path}，错误: {e}")
        return None, None

def yolo_to_voc_bbox(x_center, y_center, width, height, img_w, img_h):
    xmin = int((x_center - width/2) * img_w)
    xmax = int((x_center + width/2) * img_w)
    ymin = int((y_center - height/2) * img_h)
    ymax = int((y_center + height/2) * img_h)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)
    return xmin, ymin, xmax, ymax

def create_xml(image_id, image_path, label_path, output_xml_path):
    img_w, img_h = get_image_size(image_path)
    if img_w is None or img_h is None:
        return False
    if not os.path.exists(label_path):
        print(f"警告：标注文件不存在 {label_path}，跳过 {image_id}")
        return False

    with open(label_path, 'r') as f:
        lines = f.readlines()

    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'VOC2007'
    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)  # 保持原文件名
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'The VOC Database'
    annotation_elem = ET.SubElement(source, 'annotation')
    annotation_elem.text = 'PASCAL VOC'
    image_elem = ET.SubElement(source, 'image')
    image_elem.text = 'flickr'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_w)
    height = ET.SubElement(size, 'height')
    height.text = str(img_h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x_center, y_center, w_norm, h_norm = map(float, parts)
        cls_id = int(cls_id)
        if cls_id >= len(classes):
            print(f"警告：类别ID {cls_id} 超出类别列表，跳过该标注")
            continue
        class_name = classes[cls_id]
        xmin, ymin, xmax, ymax = yolo_to_voc_bbox(x_center, y_center, w_norm, h_norm, img_w, img_h)
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = class_name
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin_elem = ET.SubElement(bndbox, 'xmin')
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, 'ymin')
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, 'xmax')
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, 'ymax')
        ymax_elem.text = str(ymax)

    xml_str = ET.tostring(annotation, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    return True

def find_image_file(image_id):
    """在所有图片目录中查找图片，返回 (found_path, subset) 或 (None, None)"""
    for subset, img_dir in image_base.items():
        if not os.path.exists(img_dir):
            continue
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(img_dir, image_id + ext)
            if os.path.exists(candidate):
                return candidate, subset
    return None, None

def main():
    # 遍历 train, val, test 三个子集（实际只是读取ID列表）
    for subset in ['train', 'val', 'test']:
        txt_path = os.path.join(VOCdevkit_path, f'VOC2007/ImageSets/Main/{subset}.txt')
        if not os.path.exists(txt_path):
            print(f"警告：{txt_path} 不存在，跳过 {subset}")
            continue
        with open(txt_path, 'r') as f:
            image_ids = [line.strip() for line in f.readlines() if line.strip()]

        for image_id in image_ids:
            xml_path = os.path.join(VOCdevkit_path, f'VOC2007/Annotations/{image_id}.xml')
            if os.path.exists(xml_path):
                continue  # 已有XML，跳过

            # 在所有目录中查找图片文件
            image_path, found_subset = find_image_file(image_id)
            if image_path is None:
                print(f"警告：找不到图片 {image_id} 在任何目录，跳过")
                continue

            # 获取对应子集的标签目录
            label_dir = label_base.get(found_subset)
            if not label_dir or not os.path.exists(label_dir):
                print(f"警告：标签目录 {label_dir} 不存在，跳过")
                continue
            label_path = os.path.join(label_dir, image_id + '.txt')
            if not os.path.exists(label_path):
                print(f"警告：标签文件 {label_path} 不存在，跳过")
                continue

            success = create_xml(image_id, image_path, label_path, xml_path)
            if success:
                print(f"已生成XML: {xml_path}")
            else:
                print(f"生成XML失败: {image_id}")

if __name__ == "__main__":
    main()