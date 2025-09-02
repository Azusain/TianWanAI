# Multi-stage Dockerfile for cam-stream backend
# Stage 1: Builder - compile the Go application
FROM ubuntu:latest AS builder

# Install Go and other build dependencies
RUN apt-get update && apt-get install -y \
    golang-go \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Go environment variables
ENV GO_VERSION=1.21
ENV GOOS=linux
ENV GOARCH=amd64
ENV CGO_ENABLED=1

# Set work directory
WORKDIR /app

# Copy go mod files first for better caching
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN go build -ldflags "-s -w" -o cam-stream-backend .

# Stage 2: Runtime - minimal runtime environment
FROM ubuntu:latest AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    fonts-dejavu-core \
    fontconfig \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# Create a non-root user for security
RUN useradd -r -s /bin/false -m -d /app appuser

# Set work directory
WORKDIR /app

# Copy the compiled binary from builder stage
COPY --from=builder /app/cam-stream-backend .

# Copy static HTML files needed by the web interface
COPY --from=builder /app/*.html ./

# Copy necessary directories
COPY --from=builder /app/web ./web
COPY --from=builder /app/assets ./assets
COPY --from=builder /app/configs ./configs

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port used by the application
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# Command to run the application
CMD ["./cam-stream-backend"]
