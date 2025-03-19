import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 加载预训练的 Faster R-CNN 模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 定义图像预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# 加载图像
image_path = 'train/new_images/1_1_60.0_B21_5.jpg'
image_path = 'train/new_images-0/1_0_30.0_B21_10.jpg'
image_path = 'train/images-1/1_200.0_20250314203236.jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)

# 将图像移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_tensor = image_tensor.to(device)
model.to(device)

# 进行预测
with torch.no_grad():
    prediction = model([image_tensor])

# 解析预测结果
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()

# 可视化结果
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

for box, label, score in zip(boxes, labels, scores):
    if score > 0.5:  # 设置置信度阈值
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f'{label}: {score:.2f}', color='blue', bbox=dict(facecolor='red', alpha=0.5))

plt.show()
