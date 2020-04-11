use sdl2::sys::SDL_Window;
use sdl2::sys::SDL_bool::SDL_FALSE;
use sdl2::sys::SDL_SYSWM_TYPE;
use sdl2::sys::{SDL_GetError, SDL_GetWindowWMInfo, SDL_SysWMinfo};
use sdl2::sys::{SDL_MAJOR_VERSION, SDL_MINOR_VERSION, SDL_PATCHLEVEL};
use std::ffi::CString;
use std::mem;
use std::os::raw::c_char;
use vulkano::instance::InstanceExtensions;

#[derive(Debug)]
pub enum ErrorType {
    Unknown,
    PlatformNotSupported,
    Generic(String),
}

pub fn required_extensions(window: &sdl2::video::Window) -> Result<InstanceExtensions, ErrorType> {
    let wm_info = get_wminfo(window.raw())?;
    let mut extensions = InstanceExtensions {
        khr_surface: true,
        ..InstanceExtensions::none()
    };
    match wm_info.subsystem {
        SDL_SYSWM_TYPE::SDL_SYSWM_X11 => extensions.khr_xlib_surface = true,
        SDL_SYSWM_TYPE::SDL_SYSWM_WAYLAND => extensions.khr_wayland_surface = true,
        SDL_SYSWM_TYPE::SDL_SYSWM_WINDOWS => extensions.khr_win32_surface = true,
        SDL_SYSWM_TYPE::SDL_SYSWM_ANDROID => extensions.khr_android_surface = true,
        _ => return Err(ErrorType::PlatformNotSupported),
    }
    Ok(extensions)
}

fn get_wminfo(window: *mut SDL_Window) -> Result<SDL_SysWMinfo, ErrorType> {
    let mut wm_info: SDL_SysWMinfo;
    unsafe {
        wm_info = mem::zeroed();
    }
    wm_info.version.major = SDL_MAJOR_VERSION as u8;
    wm_info.version.minor = SDL_MINOR_VERSION as u8;
    wm_info.version.patch = SDL_PATCHLEVEL as u8;
    unsafe {
        if SDL_GetWindowWMInfo(window, &mut wm_info as *mut SDL_SysWMinfo) == SDL_FALSE {
            let error = CString::from_raw(SDL_GetError() as *mut c_char);
            match error.into_string() {
                Ok(x) => return Err(ErrorType::Generic(x)),
                Err(_) => return Err(ErrorType::Unknown),
            }
        }
    }
    Ok(wm_info)
}
