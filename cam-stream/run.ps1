#!/usr/bin/env pwsh

Write-Host " Starting Multi-Camera Stream Platform..." -ForegroundColor Green

# Find Go executable
$go = $null
if (Get-Command go -ErrorAction SilentlyContinue) {
    $go = "go"
} elseif (Test-Path "C:\Go\bin\go.exe") {
    $go = "C:\Go\bin\go.exe"
} elseif (Test-Path "C:\Program Files\Go\bin\go.exe") {
    $go = "C:\Program Files\Go\bin\go.exe"
} else {
    Write-Host "Go not found! Please install Go or add it to PATH" -ForegroundColor Red
    exit 1
}

Write-Host " Using Go: $go" -ForegroundColor Cyan
Write-Host " Server will start on: http://localhost:3001" -ForegroundColor Yellow
Write-Host ""

& $go run .
