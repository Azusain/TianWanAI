fn main() {
    // Skip vswhom usage on Windows to avoid C++ linking issues
    std::env::set_var("VSWHERE", "false");
    std::env::set_var("VSWHOM_DISABLE", "1");
    
    tauri_build::build()
}
