# 天空红外小目标检测数据集  | 红外小目标 天空目标检测 无人机检测 鸟类识别 低空安防9099期

## 数据集概述

本数据集专注于**天空背景下的红外小目标检测与分类**，服务于低空安防、反无人机系统及航空目标监测。数据涵盖飞机、无人机、直升机与鸟类四类空中目标，适配红外成像条件下的目标识别、空域态势感知及鸟击防范等应用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/856f85dfeee844d69922a0ebca8e68e3.png)

## 数据集核心信息

| 数据要素 | 详细说明 |
| :--- | :--- |
| **类别** | 共 **4** 类：`飞机`、`无人机`、`直升机`、`鸟` |
| **数量** | **8362** 张图像 |
| **格式** | 图像为 JPG/PNG；标签为 YOLO 格式的 txt 文件（水平边界框） |
| **应用价值** | 1. **低空安防**：支撑机场、重要设施周边的空中目标入侵检测与分类。<br>2. **反无人机**：识别违规升空飞行器，辅助反制决策。<br>3. **鸟击防范**：区分鸟类与飞行器，降低机场鸟击风险。<br>4. **算法研发**：为红外小目标检测、相似目标区分等视觉任务提供标准化训练与评测数据。 |

## 完整类别列表（共4类）

| 序号 | 英文名称 | 中文释义 |
|:---:|:---|:---|
| 0 | Plane | 飞机 |
| 1 | Drone | 无人机 |
| 2 | Helicopter | 直升机 |
| 3 | Bird | 鸟类 |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e07a8b066c042088511d1ad235ceab8.png)

## 数据三要素解析

**类别与目标覆盖**
*   数据集定义 **4** 类空中目标：**飞机**、**无人机**、**直升机**、**鸟**，覆盖了低空监测场景中常见的人造飞行器与生物目标。
*   鸟类与无人机的区分是低空安防中的经典难题，本数据集为训练高判别力模型提供了关键数据。
*   类别设置兼顾军事/民用航空器与生态保护对象，具有较强的应用综合性。

**数据规模与模态**
*   图像总量约 **8362** 张，规模充足，能支撑红外小目标检测模型的稳健训练。
*   数据为**红外成像**，不依赖外部光照，适用于夜间、雾霾等低能见度环境下的目标检测。
*   红外图像中目标呈高亮热特征，但纹理细节少、目标尺度小，对检测算法提出较高要求。

**格式与配套资源**
*   采用通用 **YOLO 格式**（txt），每张图像对应一个同名标签文件，包含类别 ID 与归一化边界框坐标，可直接接入 YOLOv8 等主流检测框架。
*   附赠完整的 **YOLOv8 训练脚本**、**评估脚本**、**推理与可视化脚本**，均带有中文注释，用户可直接运行。
*   提供 **data.yaml 配置文件示例** 与 **数据集目录结构说明**，便于快速上手。

## 知识普及：红外小目标检测的技术要点

*   **小目标挑战**：红外图像中，远距离飞机、无人机等目标像素占比极小（可能仅占图像面积的0.1%以下），常规检测模型易漏检。需采用高分辨率输入、多尺度特征融合或专门的小目标检测头。
*   **红外成像特点**：红外图像依赖物体自身热辐射成像，不依赖外部光源，适合夜间与低能见度环境。但分辨率较低、纹理细节少，目标轮廓可能模糊。
*   **类间区分**：飞机、无人机、直升机与鸟类在红外图像中形态相似，尤其远距离时难以区分。模型需依赖轮廓、尺寸、飞行姿态等特征进行判别。
*   **数据增强策略**：针对小目标与相似目标区分，可采用Mosaic、MixUp等增强策略，增加模型对复杂场景的适应能力。
*   **应用闭环**：检测结果可结合雷达、射频信号，构建多传感器融合的低空安防系统，实现"发现-识别-反制"的闭环流程。

## 快速开始（代码示例）

**1. 准备配置文件 (`data.yaml`)**
```yaml
train: ./dataset/images/train/
val: ./dataset/images/val/
nc: 4
names: ['Plane', 'Drone', 'Helicopter', 'Bird']
```

**2. 启动 YOLOv8 训练**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(
    data='./path/to/data.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    project='./runs/detect',
    name='sky_infrared_target_detection',
    optimizer='SGD',
    device='0',
    save=True,
    cache=True
)
```

**3. 模型评估**
```python
from ultralytics import YOLO

model = YOLO('./runs/detect/sky_infrared_target_detection/weights/best.pt')
metrics = model.val(data='./path/to/data.yaml')
print(metrics.box.map)
```

**4. 推理与可视化**
```python
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO('./runs/detect/sky_infrared_target_detection/weights/best.pt')

def detect_sky_objects(image_path):
    results = model.predict(source=image_path)
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes.numpy()
        for box in boxes:
            r = box.xyxy
            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            label = result.names[int(box.cls)]
            confidence = box.conf
            if confidence > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

result_image = detect_sky_objects('your_test_image.jpg')
Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).show()
```

## 标签
`红外小目标` `天空目标检测` `无人机检测` `鸟类识别` `低空安防` `目标检测` `YOLO格式` `深度学习数据集` `计算机视觉`
