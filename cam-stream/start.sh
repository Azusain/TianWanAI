#!/bin/bash
export PATH=/usr/local/go/bin:/usr/bin:$PATH
export CGO_ENABLED=1
go build -o cam-stream && ./cam-stream
