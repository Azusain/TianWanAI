# development script for unified tool app

Write-Host "starting unified tool app in development mode..." -ForegroundColor Green

# check if rust is installed
try {
    $rustVersion = cargo --version
    Write-Host "found rust: $rustVersion" -ForegroundColor Blue
} catch {
    Write-Host "error: rust not found. please install rust from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# check if tauri cli is installed
try {
    $tauriVersion = cargo tauri --version
    Write-Host "found tauri: $tauriVersion" -ForegroundColor Blue
} catch {
    Write-Host "installing tauri cli..." -ForegroundColor Yellow
    cargo install tauri-cli
}

# run the development server
Write-Host "launching development server..." -ForegroundColor Green
cargo tauri dev
