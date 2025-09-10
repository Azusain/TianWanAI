# syntax=docker/dockerfile:1.6

# ---- opencv builder ----
FROM debian:bookworm AS opencv-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.8.0

# base build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    ca-certificates \
    unzip \
    # image codecs (minimal)
    libjpeg-dev libpng-dev \
    # minimal FFmpeg for RTSP decoding only
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    # threading only
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src

# fetch opencv only (no contrib for minimal build)
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git

# configure and build
RUN cmake -S opencv -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=OFF \
    -D WITH_GTK=OFF \
    -D WITH_QT=OFF \
    -D WITH_TBB=ON \
    -D WITH_IPP=OFF \
    -D WITH_OPENMP=ON \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && ldconfig

# verify pc file exists
RUN test -f /usr/local/lib/pkgconfig/opencv4.pc


# ---- go builder (with opencv from previous stage) ----
FROM golang:1.23.4-bookworm AS go-builder

# copy opencv runtime + headers into this stage
COPY --from=opencv-builder /usr/local /usr/local

# minimal system libs for cgo link checks (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libtbb12 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

ENV CGO_ENABLED=1 \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}

# set up workdir and copy module
WORKDIR /app
COPY go.mod go.sum ./

# ensure modules are cached
RUN go mod download

# copy the rest of the source
COPY . .

# build
RUN go build -o /app/bin/cam-stream ./


# ---- runtime ----
FROM debian:bookworm-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# ultra-minimal runtime deps (headless server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    libjpeg62-turbo \
    libpng16-16 \
    libavcodec59 \
    libavformat59 \
    libavutil57 \
    libswscale6 \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# copy opencv runtime from builder
COPY --from=opencv-builder /usr/local /usr/local

# update library cache so opencv libs are found
RUN ldconfig

# copy app binary and static assets
WORKDIR /app
COPY --from=go-builder /app/bin/cam-stream /app/cam-stream
COPY --from=go-builder /app/assets /app/assets
COPY --from=go-builder /app/camera_management.html /app/
COPY --from=go-builder /app/image_viewer.html /app/

# runtime env for FFmpeg logging noise suppression and timezone
ENV AV_LOG_LEVEL=error \
    AV_LOG_FORCE_NOCOLOR=1 \
    TZ=Asia/Shanghai

# create writable dirs
RUN mkdir -p /app/output /app/_data
VOLUME ["/app/output", "/app/_data"]

EXPOSE 8080

ENTRYPOINT ["/app/cam-stream"]
