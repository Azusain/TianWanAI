# syntax=docker/dockerfile:1.6

# ---- Go builder with CGO enabled for FFmpeg integration ----


FROM golang:1.23-alpine AS go-builder

RUN apk add --no-cache \
    git \
    build-base \
    pkgconfig \
    ffmpeg-dev

# set up workdir and copy module
WORKDIR /app
COPY go.mod go.sum ./

# ensure modules are cached
RUN go mod download

# copy the rest of the source
COPY . .

# build binary with CGO enabled for FFmpeg integration
RUN CGO_ENABLED=1 GOOS=linux go build -o /app/bin/cam-stream ./


# ---- runtime ----
FROM alpine:3.18 AS runtime

# minimal runtime deps - FFmpeg tools and C++ runtime libraries
RUN apk add --no-cache \
    ffmpeg \
    ca-certificates \
    tzdata

# set timezone for Alpine Linux - more robust approach
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# copy app binary and static assets
WORKDIR /app
COPY --from=go-builder /app/bin/cam-stream /app/cam-stream
COPY --from=go-builder /app/assets /app/assets
# copy HTML templates directory
COPY --from=go-builder /app/templates /app/templates
# no need to copy separate stream detector as it's now built into the Go binary

# runtime env for FFmpeg logging noise suppression
ENV AV_LOG_LEVEL=error \
    AV_LOG_FORCE_NOCOLOR=1

# create writable dirs
RUN mkdir -p /app/output /app/_data
VOLUME ["/app/output", "/app/_data"]

EXPOSE 8080

ENTRYPOINT ["/app/cam-stream"]
