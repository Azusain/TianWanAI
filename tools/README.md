# Tianwan 工具集合

本目录包含各种用于数据处理、模型训练和评估的工具脚本。

## 工具列表

### 数据处理工具

- **[upper_body_extractor.py](#上半身裁剪工具)** - 从视频和图像中提取上半身区域
- **[dataset_splitter.py](#数据集拆分工具)** - 按比例拆分数据集为训练集和验证集
- **extract_frames_lossless.py** - 无损提取视频帧
- **frame-extraction.py** - 视频帧提取工具
- **visualize_dataset.py** - 数据集可视化工具
- **split_train_test.py** - 拆分训练和测试数据集
- **dataset_statistics.py** - 统计数据集信息

### AI 服务工具

- **ai_service.py** - AI 服务接口

### 特定任务工具

- **extract_mouse3_frames.py** / **extract_mouse3_final.py** - 鼠标数据提取工具
- **split_mouse_dataset_v6.py** - 鼠标数据集拆分

---

# 上半身裁剪工具

## 简介
从视频或图像中自动提取上半身图片，基于YOLO姿态检测识别人体关键点，裁剪肩膀到髋部区域。

## 快速使用

### 1. 激活环境
```bash
../Scripts/activate
```

### 2. 处理单个视频
```bash
python upper_body_extractor.py "视频路径.mp4"
```

### 3. 批量处理文件夹（视频）
```bash
python upper_body_extractor.py "文件夹路径" --batch
```

### 4. 批量处理图像文件夹
```bash
python upper_body_extractor.py "图像文件夹路径" --batch --images
```

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `input` | - | 视频文件或文件夹路径 |
| `--model` | `../models/yolo11m-pose.pt` | YOLO姿态模型路径 |
| `--output` / `-o` | `upper_body_crops` | 输出目录 |
| `--interval` / `-i` | `30` | 帧间隔（每30帧处理1帧）|
| `--max-frames` | `100` | 每个视频最多输出图片数 |
| `--batch` / `-b` | - | 批量处理文件夹模式 |
| `--images` | - | 处理图像文件（而非视频）|

## 使用示例

### 处理单个视频（默认参数）
```bash
python upper_body_extractor.py "video.mp4"
```
- 输出：`upper_body_crops/` 文件夹
- 每30帧提取1张，最多100张图片

### 批量处理（自定义参数）
```bash
python upper_body_extractor.py "D:\videos" -b -i 15 --max-frames 50 -o "my_output"
```
- 处理文件夹内所有视频
- 每15帧提取1张
- 每个视频最多50张图片
- 输出到 `my_output/` 文件夹

### 处理视频文件夹
```bash
python upper_body_extractor.py "D:\azutemp\tianwan_dataset\videos" -b
```

### 处理图像文件夹
```bash
python upper_body_extractor.py "D:\azutemp\tianwan_dataset\images" -b --images
```

## 输出文件命名
```
upper_body_{视频名}_frame{帧号}_person{人员编号}.jpg
```
示例：`upper_body_video1_frame000030_person0.jpg`

## 支持格式
- 视频：MP4, AVI, MOV, MKV
- 图像：JPG, JPEG, PNG, BMP
- 输出：JPG图片

## 注意事项
- 需要激活 tianwan 虚拟环境
- 确保模型文件 `yolo11m-pose.pt` 存在
- 输出目录会自动创建
- 太小的人体区域会被过滤（小于50x50像素）

---

# 数据集拆分工具

## 简介
按照指定比例（默认4:1）将数据集拆分为训练集和验证集。适用于图像分类数据集，自动创建类别对应的验证集文件夹。

## 使用方法

### 基本用法
```bash
python dataset_splitter.py "数据集路径" -o "输出路径"
```

### 自定义拆分比例
```bash
python dataset_splitter.py "数据集路径" -o "输出路径" -r 0.7
```
- `-r 0.7` 表示 70% 训练集，30% 验证集

### 指定特定类别
```bash
python dataset_splitter.py "数据集路径" -o "输出路径" --classes class1 class2
```

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `input` | - | 输入数据集路径 |
| `--output` / `-o` | - | 输出路径（必需） |
| `--ratio` / `-r` | `0.8` | 训练集比例（默认0.8为4:1） |
| `--classes` | 全部 | 指定要处理的类别（空格分隔） |
| `--seed` | `42` | 随机种子，保证结果可重现 |

## 输出说明

假设输入数据集有 class1 和 class2 两个类别文件夹，输出将包含：

- `class1/` - 80% 的 class1 图像（训练集）
- `class1_val/` - 20% 的 class1 图像（验证集）
- `class2/` - 80% 的 class2 图像（训练集）
- `class2_val/` - 20% 的 class2 图像（验证集）

## 支持格式
- 图像：JPG, JPEG, PNG, BMP, TIFF, WEBP
