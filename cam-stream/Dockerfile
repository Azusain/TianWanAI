# syntax=docker/dockerfile:1.6

# ---- go builder (pure golang, no opencv) ----
FROM golang:1.23.4-bookworm AS go-builder

# set up workdir and copy module
WORKDIR /app
COPY go.mod go.sum ./

# ensure modules are cached
RUN go mod download

# copy the rest of the source
COPY . .

# build static binary (disable CGO since we don't need opencv)
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /app/bin/cam-stream ./


# ---- runtime ----
FROM debian:bookworm-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# minimal runtime deps - only FFmpeg tools for proxy streaming
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# copy app binary and static assets
WORKDIR /app
COPY --from=go-builder /app/bin/cam-stream /app/cam-stream
COPY --from=go-builder /app/assets /app/assets
# copy HTML templates directory
COPY --from=go-builder /app/templates /app/templates

# runtime env for FFmpeg logging noise suppression and timezone
ENV AV_LOG_LEVEL=error \
    AV_LOG_FORCE_NOCOLOR=1 \
    TZ=Asia/Shanghai

# create writable dirs
RUN mkdir -p /app/output /app/_data
VOLUME ["/app/output", "/app/_data"]

EXPOSE 8080

ENTRYPOINT ["/app/cam-stream"]
