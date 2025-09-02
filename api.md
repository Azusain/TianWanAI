# 告警平台请求接口

## 请求示例

```json
{
    "image": "base64_encoded_image",
    "request_id": "2058ed71-6b61-4f66-83e6-59a31ba87d7b",
    "model": "model-name-v1.0",
    "camera_kks": "1M2DTW345TV",
    "score": 0.95,
    "x1": 0.1,
    "y1": 0.2,
    "x2": 0.4,
    "y2": 0.6,
    "timestamp": "2024-01-01T12:00:00+08:00"
}
```

## 请求字段

| 字段名 | 数据类型 | 字段解释 |
|--------|----------|----------|
| image | string | Base64编码的图片数据 |
| request_id | string | 唯一请求标识符 |
| model | string | 模型名称和版本 |
| camera_kks | string | 摄像头KKS编码 |
| score | float | 置信度分数 (0-1) |
| x1 | float | 检测框左上角x坐标 (归一化) |
| y1 | float | 检测框左上角y坐标 (归一化) |
| x2 | float | 检测框右下角x坐标 (归一化) |
| y2 | float | 检测框右下角y坐标 (归一化) |
| timestamp | string | 请求时间戳（上海时间） |

