import torch

# from torchsummary import summary  # 注释掉，不再使用
from nets.yolo import YoloBody
from thop import clever_format, profile

if __name__ == "__main__":
    input_shape = [416, 416]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 4  # 根据您的数据集修改

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YoloBody(anchors_mask, num_classes).to(device)
    # summary(m, (3, input_shape[0], input_shape[1]))  # 注释掉此行

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(m, (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2 是因为 profile 默认只计算乘法，加法则忽略
    #   若按常规将乘加都算作 FLOPs，则乘 2（参考 YOLOX 做法）
    # --------------------------------------------------------#
    flops = flops
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Total GFLOPS: {flops}")
    print(f"Total params: {params}")

# TODO： Yolov4-attention模型
# Total GFLOPS: 114.725G
# Total params: 84.592M

# Total GFLOPS: 29.987G
# Total params: 63.954M
