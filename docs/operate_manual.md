# 操作手册
1. 先下载所有的 .7z 镜像
2. 传输所有镜像到服务器 /home 下
3. 使用指令
```bash
# 查看当前所有正在运行的容器
docker ps -a

# 根据容器 ID 关闭当前正在运行的所有容器
docker stop <容器 ID 1> <容器 ID 2> <容器 ID N ...>

# 批量解压 7z 文件，得到 .tar 文件
7z x <文件 1> <文件 2> <文件 N ...>

# 清空 docker 所有的镜像和容器（危险，实际上图方便才这么做）
# 之后终端交互输入 y 确认
docker system prune -a

# 加载所有镜像
# 加载好之后可以用 docker images 检查
docker load -i <镜像文件 1.tar> <镜像文件 2.tar> <镜像文件 N.tar>

# 启动服务
# tianwan 1 （手势，老鼠，香烟，烟雾，摔倒，短袖，积水）
docker run -d -p 8901:8080 --gpus all azusaing/tianwan:latest

# tianwan 2 （安全帽，安全带，火焰）
docker run -d -p 8902:8080 --gpus all azusaing/tianwan2:latest

# 摄像头平台
docker run -d -p 8080:8080 azusaing/cam-stream:latest 
# 或者用这个可以现场采集图片和标签
# docker run -d -p 8080:8080 -e DEBUG=1 azusaing/cam-stream:latest 

# 全部加载好之后可以用 docker ps -a 检查

```



### 其他指令

```bash
# 查看容器日志
docker logs -f <容器 ID>

# 拷贝原图和标签
docker cp <摄像头平台容器 ID>:/app/debug .

# 拷贝标注图像
docker cp <摄像头平台容器 ID>:/app/root .

# 直接进入容器内部
docker exec -it <摄像头平台容器 ID> /bin/sh

```

