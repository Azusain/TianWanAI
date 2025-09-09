# 图像分类测试工具

## 简介

这个工具用于可视化测试 `tshirt_cls_v2` 模型的分类效果。它会加载指定文件夹中的图片，使用模型进行分类，并在图片上显示预测结果。

## 模型类别说明

根据你的说明，模型的类别映射为：
- **0: other** (其他类型)
- **1: tshirt** (短袖T恤)

## 使用方法

### 基本用法

```bash
cd test
python image_classifier_test.py "图片文件夹路径"
```

### 指定模型路径

```bash
python image_classifier_test.py "图片文件夹路径" --model "../models/tshirt_cls/weights/tshirt_cls_v2.pt"
```

### 自定义显示窗口大小

```bash
python image_classifier_test.py "图片文件夹路径" --size 1024 768
```

## 操作说明

工具会打开一个窗口显示图片和分类结果：

- **绿色背景**: 预测为 tshirt (类别1)
- **红色背景**: 预测为 other (类别0)
- **显示信息**: 类别名称、类别编号、置信度

### 按键操作

- **'n'键 或 空格**: 下一张图片
- **'p'键**: 上一张图片  
- **'q'键 或 ESC**: 退出测试
- **任意其他键**: 重新显示当前图片

## 输出信息

工具会在控制台显示：
1. **模型加载信息**: 设备类型、类别映射
2. **逐张处理信息**: 文件名、预测结果、置信度
3. **最终统计**: 各类别数量、分类比例

## 测试示例

假设你要测试验证集：

```bash
python image_classifier_test.py "D:\azutemp\tianwan_dataset\tshirt\tshirt_cls_v2_split\tshirt_val"
```

或者测试其他文件夹：

```bash
python image_classifier_test.py "D:\azutemp\tianwan_dataset\tshirt\tshirt_cls_v2_split\other_val"
```

## 注意事项

1. 需要激活 tianwan 虚拟环境
2. 确保模型文件存在
3. 支持的图片格式：JPG, JPEG, PNG, BMP, TIFF, WEBP
4. 按文件名排序显示图片
5. 实时显示处理进度和统计信息
