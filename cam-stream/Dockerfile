
FROM golang:1.25.1-alpine3.22 AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build static binary
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags "-s -w -extldflags '-static'" -o cam-stream-backend .

# Runtime stage - minimal Alpine
FROM alpine:latest

# Install only essential runtime dependencies
RUN apk add --no-cache ca-certificates curl ffmpeg

# Create non-root user
RUN adduser -D -s /bin/sh appuser

WORKDIR /app

# Copy binary and assets
COPY --from=builder /app/cam-stream-backend .
COPY --from=builder /app/*.html ./
COPY --from=builder /app/assets ./assets
COPY --from=builder /app/configs ./configs

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080


CMD ["./cam-stream-backend"]