# build script for unified tool app

Write-Host "building unified tool app for release..." -ForegroundColor Green

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

# build for release
Write-Host "building release version..." -ForegroundColor Green
cargo tauri build

# check if build was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "build completed successfully!" -ForegroundColor Green
    Write-Host "executable should be in: target/release/bundle/" -ForegroundColor Blue
} else {
    Write-Host "build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
