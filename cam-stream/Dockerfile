# syntax=docker/dockerfile:1.6

# ---- opencv builder ----
FROM ubuntu:22.04 AS opencv-builder

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
    # image codecs
    libjpeg-dev libpng-dev libtiff-dev \
    # video/ffmpeg/gstreamer
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    # highgui and gui backends
    libgtk-3-dev \
    # linear algebra / threading
    libtbb-dev libopenblas-dev liblapacke-dev \
    # misc
    libxvidcore-dev libx264-dev libv4l-dev \
    python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src

# fetch opencv and contrib at specific version
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git \
 && git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

# configure and build
RUN cmake -S opencv -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/src/opencv_contrib/modules \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_TBB=ON \
    -D WITH_IPP=ON \
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

# required system libs for cgo link checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libgtk-3-0 \
    libtbb12 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV CGO_ENABLED=1 \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}

# set up workdir and copy module
WORKDIR /app
COPY go.mod go.sum ./

# ensure modules are cached
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# copy the rest of the source
COPY . .

# build
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o /app/bin/cam-stream ./


# ---- runtime ----
FROM debian:bookworm-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# minimal runtime deps for OpenCV highgui/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-0 \
    libtbb12 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    ffmpeg \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# copy opencv runtime from builder
COPY --from=opencv-builder /usr/local /usr/local

# copy app binary and static assets
WORKDIR /app
COPY --from=go-builder /app/bin/cam-stream /app/cam-stream
COPY --from=go-builder /app/assets /app/assets
COPY --from=go-builder /app/camera_management.html /app/
COPY --from=go-builder /app/image_viewer.html /app/
COPY --from=go-builder /app/alerts.html /app/

# runtime env for ffmpeg logging noise suppression used by the app
ENV OPENCV_FFMPEG_LOGLEVEL=24 \
    AV_LOG_FORCE_NOCOLOR=1 \
    FFREPORT=level=error

# create writable dirs
RUN mkdir -p /app/output /app/_data
VOLUME ["/app/output", "/app/_data"]

EXPOSE 3000

ENTRYPOINT ["/app/cam-stream"]
