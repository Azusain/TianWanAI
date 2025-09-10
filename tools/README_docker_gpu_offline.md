# CentOS 7.6 离线安装 Docker GPU 支持

## 前置条件
- CentOS 7.6 已安装 NVIDIA 驱动（`nvidia-smi` 可用）
- Docker 已安装并运行

## 在线准备（联网机器执行）

```bash
# 添加 NVIDIA 仓库
sudo yum install -y yum-utils
sudo curl -s -L https://nvidia.github.io/nvidia-docker/centos7/nvidia-docker.repo \
  -o /etc/yum.repos.d/nvidia-docker.repo

# 下载 RPM 包
mkdir -p rpms && cd rpms
yumdownloader --resolve nvidia-container-toolkit nvidia-container-runtime \
  libnvidia-container-tools libnvidia-container1 libseccomp

# 打包
tar -czf nvidia-docker-centos7.tar.gz *.rpm
```

## 离线安装（目标机器执行）

```bash
# 解压并安装
tar -xzf nvidia-docker-centos7.tar.gz
sudo yum localinstall -y *.rpm

# 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 使用

```bash
# 运行 GPU 容器
docker run --gpus all your-gpu-image
```

## 验证

```bash
docker info | grep -i runtime
```

应显示包含 `nvidia` 的 runtime 配置。
