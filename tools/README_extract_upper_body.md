# Upper Body Extraction Tool

这个工具可以从视频中自动提取人体的上半身区域，使用 YOLO pose 检测来精确定位肩膀和腰部关键点。

## 功能特性

- 📹 支持各种视频格式 (mp4, avi, mov, etc.)
- 🎯 基于人体关键点精确裁剪上半身区域
- ⚡ GPU 加速推理
- 🎚️ 可控制抽帧频率
- 🐛 可选择保存调试帧（带关键点标注）
- 📊 详细的处理统计信息

## 快速开始

### 基本用法

```bash
# 处理视频，每30帧抽取一次
python tools/extract_upper_body.py video.mp4

# 处理视频，每10帧抽取一次
python tools/extract_upper_body.py video.mp4 --interval 10

# 自定义输出目录
python tools/extract_upper_body.py video.mp4 --output my_crops
```

### 高级用法

```bash
# 完整参数示例
python tools/extract_upper_body.py video.mp4 \
    --pose-model models/yolo11m-pose.pt \
    --output upper_body_crops \
    --interval 30 \
    --confidence 0.7 \
    --prefix tshirt_data \
    --debug \
    --margin 0.15
```

## 参数说明

| 参数 | 短参数 | 默认值 | 说明 |
|------|-------|--------|------|
| `video` | - | - | 输入视频文件路径 |
| `--pose-model` | - | `models/yolo11m-pose.pt` | pose 检测模型路径 |
| `--output` | `-o` | `upper_body_crops` | 输出目录 |
| `--interval` | `-i` | `30` | 抽帧间隔（1=每帧，30=每30帧） |
| `--confidence` | `-c` | `0.5` | 人体检测最小置信度 |
| `--prefix` | `-p` | `upper_body` | 保存图片的文件名前缀 |
| `--debug` | - | `False` | 保存带关键点标注的调试帧 |
| `--margin` | `-m` | `0.1` | 裁剪边距比例 |

## 输出文件

### 裁剪图片命名规则
```
{prefix}_{video_name}_frame{frame_number}_person{person_index}.jpg
```

例如：`upper_body_myvideo_frame000030_person0.jpg`

### 目录结构
```
upper_body_crops/
├── upper_body_myvideo_frame000030_person0.jpg
├── upper_body_myvideo_frame000060_person0.jpg
├── upper_body_myvideo_frame000060_person1.jpg
└── debug_frames/  (if --debug enabled)
    ├── debug_myvideo_frame000030.jpg
    └── debug_myvideo_frame000060.jpg
```

## 使用案例

### 1. 为短袖分类器收集数据
```bash
# 每15帧抽取一次，高置信度，保存调试帧
python tools/extract_upper_body.py training_video.mp4 \
    --interval 15 \
    --confidence 0.8 \
    --prefix tshirt_train \
    --debug
```

### 2. 快速浏览视频内容
```bash
# 每60帧抽取一次（约每2秒）
python tools/extract_upper_body.py content_video.mp4 \
    --interval 60 \
    --prefix preview
```

### 3. 高精度数据收集
```bash
# 每5帧抽取，大边距，低置信度阈值
python tools/extract_upper_body.py high_quality_video.mp4 \
    --interval 5 \
    --margin 0.2 \
    --confidence 0.3 \
    --prefix high_quality
```

## 技术细节

### 关键点检测
使用 YOLO11m-pose 模型检测 17 个 COCO 格式关键点：
- 肩膀：索引 5（左肩），6（右肩）
- 腰部：索引 11（左髋），12（右髋）

### 裁剪逻辑
1. 找到肩膀最高点（y值最小）
2. 找到腰部最低点（y值最大）
3. 使用人体 bbox 的 x 边界
4. 添加指定比例的边距
5. 确保边界在图像范围内

### 性能优化
- GPU 自动检测和使用
- 模型预热减少首次推理延迟
- 批处理推理（如果需要可以扩展）

## 故障排除

### 常见问题

**Q: 提示找不到 pose 模型？**
A: 确保 `models/yolo11m-pose.pt` 文件存在，或使用 `--pose-model` 参数指定正确路径。

**Q: 裁剪的上半身区域太小？**
A: 尝试增加 `--margin` 参数值，例如 `--margin 0.2`。

**Q: 处理速度太慢？**
A: 增加 `--interval` 参数值来减少处理的帧数，或确保使用 GPU。

**Q: 检测不到人？**
A: 降低 `--confidence` 参数值，例如 `--confidence 0.3`。

## 依赖要求

- Python 3.8+
- ultralytics
- opencv-python
- torch
- loguru
- numpy

## 性能参考

在 RTX 4090 上的大致性能：
- 1080p 视频：约 15-20 FPS
- 每处理 1000 帧约需 1-2 分钟
- 内存占用：约 2-4 GB
