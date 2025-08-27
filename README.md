### TianWan


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
  docker run -d -p 8902:8080 --gpus '"device=1"' --cpus=16 -e NPROC=6 -e MODEL="gesture" azusaing/tianwan:1.0.0
  
  # test on cpu
  docker run -d --rm -p 8902:8080 -e NPROC=6 -e MODEL="gesture" azusaing/tianwan:1.0.0
```

### Tested Interface

- gesture
- mouse
- ponding
- smoke
- tshirt
- fall

### Notes

```bash
  # Dockerfile_Cuda_11_1 is not applicable to this repo, 
  # but it can be used for other projects.
```