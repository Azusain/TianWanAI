### 田湾核电项目


#### Notes

- GESTURE = "gesture"
- PONDING = "ponding"
- SMOKE = "smoke"
- TSHIRT = "tshirt"
- MOUSE = "mouse"
- FALL = "fall"



### BUILD

```bash
  docker build -t service:latest .
```

### RUN

```bash
  # gpu 
  docker run -d -p 8902:8902 --gpus '"device=1"' --cpus=16 -e NPROC=6 -e MODEL="gesture" tianwan
  
  # test on cpu
  docker run -p 8902:8902 -e NPROC=6 -e MODEL="gesture" tianwan
```

### Tested Interface

- gesture
- mouse
- ponding
- smoke
- tshirt
- fall

