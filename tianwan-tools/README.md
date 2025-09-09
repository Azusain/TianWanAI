# Tianwan Tools Backend

一个基于 Rust 的 AI 工具后端服务，提供各种 AI 相关功能的 REST API。

## 功能特性

- 🔥 基于 Axum 的高性能 Web 服务器
- 🧠 AI 工具服务（文本生成、图像分析、代码补全、翻译）
- 🚀 异步处理支持
- 📝 结构化日志记录
- 🌐 CORS 支持
- 🔧 Python 集成的 AI 服务脚本

## API 端点

### 健康检查
```
GET /health
```

### 获取可用工具列表
```
GET /api/v1/tools
```

### 文本生成
```
POST /api/v1/generate-text
Content-Type: application/json

{
  "prompt": "你的提示内容",
  "max_tokens": 100
}
```

## 快速开始

### 环境要求

- Rust 1.70+
- Python 3.8+（用于 AI 服务脚本）
- Cargo

### 安装和运行

1. 构建项目：
```bash
cargo build --release
```

2. 运行服务器：
```bash
cargo run
```

服务器将在 `http://0.0.0.0:3000` 上启动。

### 测试 API

测试健康检查：
```bash
curl http://localhost:3000/health
```

测试工具列表：
```bash
curl http://localhost:3000/api/v1/tools
```

测试文本生成：
```bash
curl -X POST http://localhost:3000/api/v1/generate-text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```
