import sys
from ultralytics.utils.ops import scale_coords, non_max_suppression
sys.path.append('yolov5')
import cv2
import torch
import random
import os
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

device = select_device('')
model = attempt_load('yolov5/runs/train/exp8/weights/best.pt', device=device).eval()

val_images_dir = 'yolov5/train/yolo/images'
val_labels_dir = 'yolov5/train/yolo/labels'

os.makedirs(val_labels_dir, exist_ok=True)

folder = val_images_dir
img_path = os.path.join(folder, random.choice(os.listdir(folder)))
print(f"检测图像: {img_path}")
print(f"模型类别映射: {model.names}")  # 应输出类似 {0: 'dice'}

img0 = cv2.imread(img_path)
if img0 is None:
    print(f"图像加载失败: {img_path}")
    sys.exit(1)

# 使用letterbox进行预处理
img_resized, ratio, pad = letterbox(img0, new_shape=640, auto=False, stride=model.stride.max())
print(f"Letterbox处理后的尺寸: {img_resized.shape[:2]}")  # 正确尺寸
print(f"缩放比例: {ratio}, 填充值: {pad}")  # 新增调试输出

# 打印ratio和pad的类型
print(f"ratio类型: {type(ratio)}, pad类型: {type(pad)}")

# 将ratio和pad转换为torch.Tensor类型
ratio = torch.tensor(ratio, device=device).float()
pad = torch.tensor(pad, device=device).float()

# 打印ratio和pad的值
print(f"ratio值: {ratio}, pad值: {pad}")

# BGR转RGB并转置为CHW格式
img = img_resized[:, :, ::-1].transpose(2, 0, 1).copy()  # 关键修改：添加.copy()
img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)

# 修改后的完整检测流程
with torch.no_grad():
    pred = model(img)[0]
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

if pred is not None and len(pred):
    # 正确传入目标尺寸参数（原图高宽）
    pred[:, :4] = scale_coords(
        img_resized.shape[:2],
        pred[:, :4].float(),
        img0.shape[:2],
        ratio_pad=(ratio, pad)
    ).round().long()

    # 添加调试输出
    print(f"缩放前检测框示例：{pred[0, :4].cpu().numpy()}")
    print(f"缩放后检测框示例：{pred[0, :4].cpu().numpy()}")

    # 筛选dice类别（根据model.names调整索引）
    dice_detections = pred[pred[:, 5].int() == 0]  # 添加.int()强制类型转换

    if len(dice_detections) > 0:
        # 找到置信度最高的检测框
        max_conf_idx = torch.argmax(dice_detections[:, 4])  # 第4列为置信度值
        best_det = dice_detections[max_conf_idx]

        # 解包最优检测结果
        *xyxy, conf, cls = best_det

        # 构建标签（保留两位小数）
        label = f'{model.names[int(cls)]} {conf.item():.2f}'  # 使用.item()获取标量值

        # 绘制最高置信度框
        cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])),
                      (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        # 添加置信度文本（调整文字位置避免超出画面）
        y_label_pos = max(int(xyxy[1]) - 10, 20)  # 确保文字不会超出图像顶部
        cv2.putText(img0, label, (int(xyxy[0]), y_label_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 打印最高置信度信息
        print(f"最优检测结果 - 置信度：{conf.item():.4f}, 位置：{list(map(int, xyxy))}")
    else:
        print("未检测到dice")
else:
    print("检测结果为空")

cv2.imshow('Detection', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_dir = 'yolov5/val/yolo/output'
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, os.path.basename(img_path))
cv2.imwrite(output_filename, img0)
print(f"标注后的图像已保存至: {output_filename}")
