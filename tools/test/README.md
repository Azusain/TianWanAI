# Cam-Stream 并发测试工具

这个工具用于测试 cam-stream 服务器的并发安全性，专门检测 nil pointer 异常和竞态条件问题。

## 功能特点

- **多线程并发测试**：同时运行 20 个线程进行随机操作
- **全面的 API 测试**：覆盖摄像头和推理服务器的 CRUD 操作
- **配置文件测试**：并发修改配置文件
- **错误分类统计**：自动识别和统计不同类型的错误
- **实时监控**：每 5 秒打印一次统计信息

## 使用方法

### 前提条件
1. 确保 cam-stream 服务器正在运行（默认端口 8080）
2. 安装 Python 依赖：`pip install requests loguru`

### 运行测试
```bash
# 进入测试目录
cd tools/test

# 运行测试（默认 60 秒）
python cam_stream_concurrency_test.py
```

### 配置选项
可以修改脚本顶部的配置常量：
- `MAX_CONCURRENT_THREADS`: 并发线程数（默认 20）
- `TEST_DURATION_SECONDS`: 测试持续时间（默认 60 秒）
- `SERVER_BASE_URL`: 服务器地址（默认 http://localhost:8080）

## 测试操作

脚本会随机执行以下操作：
- **摄像头操作**：创建、更新、删除、查询摄像头
- **推理服务器操作**：创建、更新、删除、查询推理服务器
- **配置文件操作**：修改 config.json 文件
- **状态查询**：获取服务器状态

每个操作都有不同的权重，读取操作更频繁，删除操作较少。

## 输出示例

```
🚀 starting cam-stream concurrency stress test
server: http://localhost:8080
threads: 20
duration: 60s

============================================================
CONCURRENCY TEST STATISTICS
============================================================
Total Requests:        1247
Successful Requests:   1198
Failed Requests:       49
Success Rate:          96.07%

ERROR ANALYSIS:
Total Errors:          49
Nil Pointer Errors:    0 🚨
Race Condition Errors: 0 ⚠️
Timeout Errors:        12

RECENT ERRORS (last 5):
  [18:25:32] timeout: timeout on GET /api/cameras
  [18:25:35] http_error: camera creation HTTP 400: invalid request
```

## 错误检测

脚本会特别关注以下错误类型：
- **Nil Pointer Errors**: 包含 "nil pointer" 或 "null pointer" 的错误
- **Race Condition Errors**: 包含 "race" 或 "concurrent" 的错误
- **Timeout Errors**: 请求超时错误
- **HTTP Errors**: HTTP 状态码错误
- **JSON Errors**: JSON 解析错误

## 压力测试场景

这个工具模拟了以下并发场景：
1. **高频读取**：多个线程同时获取摄像头列表
2. **并发写入**：多个线程同时创建/更新配置
3. **混合操作**：同时进行读写删除操作
4. **配置冲突**：多个线程同时修改配置文件
5. **资源竞争**：对同一摄像头进行并发操作

## 停止测试

- 按 `Ctrl+C` 可以随时停止测试
- 测试结束后会显示完整的统计信息
- 如果发现 nil pointer 或竞态条件错误，会特别标注

## 故障排除

如果遇到连接错误：
1. 检查 cam-stream 服务器是否正在运行
2. 确认端口 8080 没有被其他程序占用
3. 检查防火墙设置

如果测试成功率很低：
1. 可能是服务器负载过高
2. 尝试减少并发线程数
3. 增加请求超时时间
