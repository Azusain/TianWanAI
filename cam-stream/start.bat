@echo off
cd /d "C:\Users\azusaing\Desktop\Code\tianwan\cam-stream"
wsl -d Ubuntu-22.04 bash -c "export PATH=\"/usr/local/go/bin:/usr/bin:$PATH\" && export CGO_ENABLED=1 && export FRAME_RATE=10 && go build -o cam-stream && ./cam-stream"
