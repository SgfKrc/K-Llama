"""
QLH 边缘推理系统 — 打包版启动器
================================
跨平台（Windows / Linux）安装包的主入口点。

双引擎架构:
  1. llama.cpp + GGUF  — CPU/集显默认引擎（Q4_K_M ~1.16 GB）
  2. PyTorch + bitsandbytes — CUDA 引擎（INT4 ~1.75 GB 显存）

启动流程:
  0. Tailscale 组网检查（首次启动引导加入）
  1. 检测 CUDA 可用性 + 模型文件
  2. 若缺失 → 弹出系统对话框引导下载（智能推荐格式）
  3. 模型就绪 → 自动选择最优引擎 → 后台启动 FastAPI（端口 8000）
  4. 自动打开系统浏览器加载 React 前端（Linux）或 pywebview 原生窗口（Windows）
"""

from __future__ import annotations

import os
import sys

# MODEL-SUPPLY-01: dispatch before logging redirection, config, backend imports,
# single-instance checks or TUI startup.  This is a fresh offline child process.
if __name__ == "__main__" and sys.argv[1:2] in (
    ["--qlh-cut-layers"], ["--qlh-convert-hf-to-gguf"],
):
    from model_supply_runtime import dispatch_helper

    raise SystemExit(dispatch_helper(sys.argv[1], sys.argv[2:]))

import ipaddress
import json

# ═══════════════════════════════════════════════════════════════════
# ★ 平台检测（影响后续分支逻辑）
# ═══════════════════════════════════════════════════════════════════
IS_LINUX = sys.platform == "linux"
IS_WINDOWS = sys.platform == "win32"

# ═══════════════════════════════════════════════════════════════════
# ★ 静默模式：PyInstaller console=False 时，重定向 stdout/stderr 到日志文件
# 必须在 import logging 之前执行，否则 basicConfig 的 StreamHandler 已绑定到原始 stderr
# ═══════════════════════════════════════════════════════════════════
def _redirect_frozen_logs():
    if getattr(sys, 'frozen', False):
        import datetime as _dt
        _redirect_dir = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "logs")
        try:
            os.makedirs(_redirect_dir, exist_ok=True)
            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            sys.stdout = open(os.path.join(_redirect_dir, f"stdout_{_ts}.log"), "w", encoding="utf-8")
            sys.stderr = open(os.path.join(_redirect_dir, f"stderr_{_ts}.log"), "w", encoding="utf-8")
        except Exception:
            pass

import logging
import threading
import time
import subprocess as _sp

_WINDOWS_NO_WINDOW = getattr(_sp, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0

import ssl  # noqa: E402, F401

# 确保 src 目录在 path 中（开发模式：launcher.py 在 packaging/ 子目录下）
# PyInstaller 打包后所有模块由 bootloader 加载，无需手动添加 path
if getattr(sys, 'frozen', False):
    # PyInstaller 模式：模块在 PYZ 归档中，Python 可正常导入
    _launcher_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    _app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _launcher_dir = os.path.join(_app_root, "bin") if os.path.isdir(os.path.join(_app_root, "bin")) else os.path.join(_app_root, "packaging", "packaging")
    _core_root = os.environ.get(
        "QLH_CORE_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    _src_dir = os.path.abspath(os.path.join(_core_root, "src"))
    if os.path.isdir(_src_dir):
        sys.path.insert(0, _src_dir)
    # 同目录也加入 path
    sys.path.insert(0, _launcher_dir)

logger = logging.getLogger("launcher")
_ACTIVE_STARTUP_SPLASH = None


def _startup_timeout_seconds() -> float:
    """Return the API startup timeout, tolerating an invalid environment value."""
    raw_value = os.environ.get("QLH_STARTUP_TIMEOUT", "60")
    try:
        return max(15.0, float(raw_value))
    except (TypeError, ValueError):
        logger.warning("忽略无效的 QLH_STARTUP_TIMEOUT=%r", raw_value)
        return 60.0


def _launcher_log_hint() -> str:
    base_dir = (
        os.path.dirname(os.path.abspath(sys.executable))
        if getattr(sys, "frozen", False)
        else os.path.abspath(os.path.join(_launcher_dir, ".."))
    )
    return os.path.join(base_dir, "logs")


def _requested_launch_mode(argv: list[str]) -> str:
    """Resolve the public application flags while rejecting ambiguous modes."""
    aliases = {
        "--ui": "ui",
        "--web": "ui",
        "--tui": "tui",
    }
    requested = {aliases[arg] for arg in argv if arg in aliases}
    if len(requested) > 1:
        raise ValueError("--ui/--web 与 --tui 不能同时使用")
    return next(iter(requested), "tui")




def _startup_icon_path() -> str:
    """Return the application icon source for the native startup window."""
    if getattr(sys, "frozen", False):
        return os.path.abspath(sys.executable)
    return os.path.join(_launcher_dir, "leds.ico")


class _StartupSplash:
    """Native Windows application startup window shared by Web UI and TUI."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and IS_WINDOWS)
        self._thread = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._status = "正在启动应用..."
        self._progress = 2
        self._hwnd = None
        self._status_hwnd = None
        self._progress_hwnd = None
        self._percent_hwnd = None
        self._wndproc = None

    @property
    def hwnd(self):
        return self._hwnd

    def snapshot(self) -> dict:
        """Return the application splash state without exposing controls."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "progress": self._progress,
                "status": self._status,
                "closed": self._closed.is_set(),
            }

    def start(self):
        if not self.enabled:
            return self
        self._thread = threading.Thread(
            target=self._run_windows,
            name="startup-splash",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=0.8)
        return self


    def update(self, progress: int, status: str) -> None:
        with self._lock:
            self._progress = max(0, min(100, int(progress)))
            self._status = str(status or "正在启动应用...")
            progress_value = self._progress
            status_value = self._status

        if not self.enabled or not self._ready.is_set():
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            user32.SendMessageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_size_t,
                ctypes.c_ssize_t,
            ]
            if self._status_hwnd:
                user32.SetWindowTextW(self._status_hwnd, status_value)
            if self._progress_hwnd:
                user32.SendMessageW(self._progress_hwnd, 0x0402, progress_value, 0)
            if self._percent_hwnd:
                user32.SetWindowTextW(self._percent_hwnd, f"{progress_value}%")
        except Exception:
            logger.debug("更新启动页失败", exc_info=True)

    def close(self) -> None:
        self._closed.set()
        if not self.enabled:
            return
        try:
            import ctypes

            if self._hwnd:
                ctypes.windll.user32.PostMessageW.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.c_size_t,
                    ctypes.c_ssize_t,
                ]
                ctypes.windll.user32.PostMessageW(self._hwnd, 0x0010, 0, 0)
        except Exception:
            logger.debug("关闭启动页失败", exc_info=True)

    def _run_windows(self) -> None:
        user32 = None
        gdi32 = None
        background = None
        icon_handle = None
        fonts = []
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            gdi32 = ctypes.windll.gdi32
            comctl32 = ctypes.windll.comctl32

            class WNDCLASSW(ctypes.Structure):
                _fields_ = [
                    ("style", wintypes.UINT),
                    ("lpfnWndProc", ctypes.c_void_p),
                    ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE),
                    ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HANDLE),
                    ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR),
                    ("lpszClassName", wintypes.LPCWSTR),
                ]

            class INITCOMMONCONTROLSEX(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("dwICC", wintypes.DWORD),
                ]

            user32.CreateWindowExW.argtypes = [
                wintypes.DWORD,
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                wintypes.DWORD,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.HWND,
                wintypes.HANDLE,
                wintypes.HINSTANCE,
                ctypes.c_void_p,
            ]
            user32.CreateWindowExW.restype = wintypes.HWND
            # ★ 2026-10-03：GDI 调用的参数是 **DC 句柄**，64 位下是 64 位值；ctypes 默认
            #   按 `c_int`（32 位）传 ⇒ 每次收到 `WM_CTLCOLOR*` 都抛
            #   `ctypes.ArgumentError: argument 1: OverflowError: int too long to convert`
            #   （实测把日志刷了十几行）。显式声明句柄与颜色类型。
            gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
            gdi32.SetBkMode.restype = ctypes.c_int
            gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
            gdi32.SetTextColor.restype = wintypes.COLORREF
            user32.DefWindowProcW.argtypes = [
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.DefWindowProcW.restype = ctypes.c_ssize_t
            user32.SendMessageW.argtypes = [
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.SendMessageW.restype = ctypes.c_ssize_t
            user32.PostMessageW.argtypes = [
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.DestroyWindow.argtypes = [wintypes.HWND]
            user32.IsWindow.argtypes = [wintypes.HWND]
            user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.UpdateWindow.argtypes = [wintypes.HWND]
            user32.SetForegroundWindow.argtypes = [wintypes.HWND]
            user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
            user32.LoadCursorW.restype = wintypes.HANDLE
            user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            gdi32.CreateSolidBrush.argtypes = [wintypes.DWORD]
            gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
            gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
            gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
            gdi32.CreateFontW.restype = wintypes.HANDLE
            user32.PrivateExtractIconsW.argtypes = [
                wintypes.LPCWSTR,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(wintypes.HICON),
                ctypes.POINTER(wintypes.UINT),
                wintypes.UINT,
                wintypes.UINT,
            ]
            user32.PrivateExtractIconsW.restype = wintypes.UINT

            controls = INITCOMMONCONTROLSEX(
                ctypes.sizeof(INITCOMMONCONTROLSEX), 0x00000020,
            )
            comctl32.InitCommonControlsEx(ctypes.byref(controls))

            # Keep the launcher visually aligned with the React/TUI dark shell.
            background = gdi32.CreateSolidBrush(0x000C0B0B)
            hinstance = kernel32.GetModuleHandleW(None)
            class_name = f"QLHStartupSplash_{os.getpid()}"

            WM_CLOSE = 0x0010
            WM_DESTROY = 0x0002
            WM_CTLCOLORBTN = 0x0135
            WM_CTLCOLORSTATIC = 0x0138
            TRANSPARENT = 1

            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )

            ui_button = None
            tui_button = None

            def window_proc(hwnd, message, wparam, lparam):
                if message == WM_CLOSE:
                    user32.DestroyWindow(hwnd)
                    return 0
                if message == WM_DESTROY:
                    user32.PostQuitMessage(0)
                    return 0
                if message in (WM_CTLCOLORSTATIC, WM_CTLCOLORBTN):
                    gdi32.SetBkMode(wparam, TRANSPARENT)
                    gdi32.SetTextColor(wparam, 0x00F5F5F5)
                    return background
                return user32.DefWindowProcW(hwnd, message, wparam, lparam)

            self._wndproc = WNDPROC(window_proc)
            window_class = WNDCLASSW()
            window_class.style = 0x0003
            window_class.lpfnWndProc = ctypes.cast(self._wndproc, ctypes.c_void_p).value
            window_class.hInstance = hinstance
            window_class.hCursor = user32.LoadCursorW(None, ctypes.c_void_p(32512))
            window_class.hbrBackground = background
            window_class.lpszClassName = class_name
            user32.RegisterClassW(ctypes.byref(window_class))

            width, height = 540, 300
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            left = max(0, (screen_width - width) // 2)
            top = max(0, (screen_height - height) // 2)
            hwnd = user32.CreateWindowExW(
                0x00040000,
                class_name,
                "QLH · BJTU 启动器",
                0x80000000 | 0x00800000,
                left,
                top,
                width,
                height,
                None,
                None,
                hinstance,
                None,
            )
            if not hwnd:
                raise ctypes.WinError()
            self._hwnd = hwnd

            def create_control(
                class_value, text, style, x, y, w, h, control_id=0,
            ):
                return user32.CreateWindowExW(
                    0,
                    class_value,
                    text,
                    0x40000000 | 0x10000000 | style,
                    x,
                    y,
                    w,
                    h,
                    hwnd,
                    control_id,
                    hinstance,
                    None,
                )

            icon_control = create_control("STATIC", "", 0x00000043, 38, 34, 72, 72)
            title_control = create_control(
                "STATIC", "QLH 边缘推理系统", 0, 128, 39, 360, 40,
            )
            subtitle_control = create_control(
                "STATIC", "轻量化大模型分布式边缘推理优化系统", 0,
                130, 82, 360, 26,
            )
            self._status_hwnd = create_control(
                "STATIC", self._status, 0, 40, 144, 455, 28,
            )
            self._progress_hwnd = create_control(
                "msctls_progress32", "", 0x00000001, 40, 181, 455, 18,
            )
            self._percent_hwnd = create_control(
                "STATIC", f"{self._progress}%", 0x00000002, 432, 208, 62, 24,
            )
            footer_control = create_control(
                "STATIC",
                "正在准备本地服务，请稍候",
                0, 40, 239, 420, 24,
            )
            title_font = gdi32.CreateFontW(
                -27, 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 5, 0,
                "Microsoft YaHei UI",
            )
            text_font = gdi32.CreateFontW(
                -16, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 5, 0,
                "Microsoft YaHei UI",
            )
            small_font = gdi32.CreateFontW(
                -14, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 5, 0,
                "Microsoft YaHei UI",
            )
            fonts = [title_font, text_font, small_font]
            user32.SendMessageW(title_control, 0x0030, title_font, True)
            for control in (
                subtitle_control,
                self._status_hwnd,
                self._percent_hwnd,
            ):
                user32.SendMessageW(control, 0x0030, text_font, True)
            for control in (ui_button, tui_button):
                if control:
                    user32.SendMessageW(control, 0x0030, text_font, True)
            user32.SendMessageW(footer_control, 0x0030, small_font, True)

            user32.SendMessageW(self._progress_hwnd, 0x0406, 0, 100)
            user32.SendMessageW(self._progress_hwnd, 0x0402, self._progress, 0)
            user32.SendMessageW(self._progress_hwnd, 0x0409, 0, 0x00F5F5F5)
            user32.SendMessageW(self._progress_hwnd, 0x2001, 0, 0x00363030)
            try:
                uxtheme = ctypes.windll.uxtheme
                for control in (ui_button, tui_button, self._progress_hwnd):
                    if control:
                        uxtheme.SetWindowTheme(control, "DarkMode_Explorer", None)
            except Exception:
                pass

            icon_handle = wintypes.HICON()
            icon_id = wintypes.UINT()
            try:
                extracted = user32.PrivateExtractIconsW(
                    _startup_icon_path(),
                    0,
                    64,
                    64,
                    ctypes.byref(icon_handle),
                    ctypes.byref(icon_id),
                    1,
                    0,
                )
            except Exception:
                extracted = 0
            if extracted and icon_handle:
                icon_value = int(icon_handle.value or 0)
                user32.SendMessageW(icon_control, 0x0170, icon_value, 0)
                user32.SendMessageW(hwnd, 0x0080, 1, icon_value)
                user32.SendMessageW(hwnd, 0x0080, 0, icon_value)

            try:
                corner = ctypes.c_int(2)
                dark_mode = ctypes.c_int(1)
                dwmapi = ctypes.windll.dwmapi
                dwmapi.DwmSetWindowAttribute.argtypes = [
                    wintypes.HWND,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                ]
                dwmapi.DwmSetWindowAttribute(
                    hwnd, 33, ctypes.byref(corner), ctypes.sizeof(corner),
                )
                dwmapi.DwmSetWindowAttribute(
                    hwnd, 20, ctypes.byref(dark_mode), ctypes.sizeof(dark_mode),
                )
            except Exception:
                pass

            user32.ShowWindow(hwnd, 5)
            user32.UpdateWindow(hwnd)
            user32.SetForegroundWindow(hwnd)
            self._ready.set()
            self.update(self._progress, self._status)
            if self._closed.is_set():
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)

            message = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))

        except Exception:
            logger.warning("原生启动页创建失败", exc_info=True)
        finally:
            try:
                if user32 is not None and self._hwnd and user32.IsWindow(self._hwnd):
                    user32.DestroyWindow(self._hwnd)
            except Exception:
                logger.debug("销毁启动页窗口失败", exc_info=True)
            self._hwnd = None
            self._status_hwnd = None
            self._progress_hwnd = None
            self._percent_hwnd = None
            try:
                if user32 is not None and icon_handle:
                    user32.DestroyIcon(icon_handle)
            except Exception:
                pass
            if gdi32 is not None:
                for font in fonts:
                    try:
                        if font:
                            gdi32.DeleteObject(font)
                    except Exception:
                        pass
                try:
                    if background:
                        gdi32.DeleteObject(background)
                except Exception:
                    pass
            self._ready.set()

# ================================================================
# Tailscale 组网配置
# ================================================================
TAILSCALE_INVITE_URL = "https://login.tailscale.com/uinv/iWAME6zVuB11wUxixU2Z611"
if IS_LINUX:
    TAILSCALE_DOWNLOAD_URL = "https://tailscale.com/download/linux"
else:
    TAILSCALE_DOWNLOAD_URL = "https://tailscale.com/download/windows"

# 配置目录（跨平台：Linux 用 XDG，Windows 用 LOCALAPPDATA）
if IS_LINUX:
    _CONFIG_DIR = os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "qlh",
    )
else:
    _CONFIG_DIR = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "QLH-Edge-Inference",
    )
_TAILSCALE_FLAG_DIR = _CONFIG_DIR
_TAILSCALE_FLAG_FILE = os.path.join(_TAILSCALE_FLAG_DIR, ".tailscale_joined")


# 对话框返回值常量（保持与 Win32 MessageBox 兼容）
_IDYES = 6
_IDNO = 7
_IDCANCEL = 2
_MB_OK = 0x00000000
_MB_OKCANCEL = 0x00000001
_MB_YESNO = 0x00000004
_MB_YESNOCANCEL = 0x00000003
_MB_ICONINFORMATION = 0x00000040
_MB_ICONQUESTION = 0x00000020
_MB_ICONWARNING = 0x00000030


def _has_interactive_stdin() -> bool:
    """判断当前进程是否有可交互 stdin。PyInstaller windowed 模式没有控制台。"""
    try:
        return sys.stdin is not None and not sys.stdin.closed and sys.stdin.isatty()
    except Exception:
        return False


def _safe_input(prompt: str = "", default: str | None = None) -> str | None:
    """input() 的安全包装，避免 windowed 打包版触发 lost sys.stdin 崩溃。"""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt, RuntimeError, OSError) as e:
        logger.warning(f"无法读取控制台输入: {e}")
        return default


def _safe_pause(message: str = "按 Enter 键退出..."):
    """仅在有控制台时暂停；无控制台时不阻塞、不崩溃。"""
    print(message)
    _safe_input(default="")


def _show_windows_messagebox(title: str, message: str,
                              flags: int = _MB_OK | _MB_ICONINFORMATION,
                              owner_hwnd=None) -> int:
    """Windows: ctypes.windll MessageBox。"""
    try:
        import ctypes
        return ctypes.windll.user32.MessageBoxW(
            owner_hwnd or 0, message, title, flags
        )
    except Exception as e:
        logger.warning(f"MessageBox 显示失败: {e}")
        return _IDCANCEL


def _show_linux_dialog(title: str, message: str,
                        buttons: str = "ok") -> int:
    """
    Linux: 使用 zenity 显示对话框，返回 Win32 兼容的返回值。

    buttons:
      "ok"       → zenity --info       → _IDCANCEL (zenity info 无返回值区分)
      "okcancel" → zenity --question   → OK=0→_IDYES, Cancel=1→_IDNO
      "yesno"    → zenity --question   → Yes=0→_IDYES, No=1→_IDNO
      "yesnocancel" → zenity --question --extra-button ... 不支持三层完美映射
    """
    import shutil
    if shutil.which("zenity"):
        try:
            if buttons == "ok":
                _sp.run(["zenity", "--info", "--title", title,
                         "--text", message, "--width=450"],
                        timeout=30)
                return _IDYES
            elif buttons in ("okcancel", "yesno"):
                rc = _sp.run(["zenity", "--question", "--title", title,
                              "--text", message, "--width=450",
                              "--ok-label=是(Y)", "--cancel-label=否(N)"],
                             timeout=30).returncode
                return _IDYES if rc == 0 else _IDNO
            elif buttons == "yesnocancel":
                # zenity 不支持三按钮；用 --extra-button 模拟
                rc = _sp.run(["zenity", "--question", "--title", title,
                              "--text", message, "--width=450",
                              "--ok-label=是(Y)", "--cancel-label=否(N)",
                              "--extra-button=取消"],
                             capture_output=True, text=True, timeout=30)
                if rc.returncode == 0:
                    # extra-button 被点击时 zenity 退出码仍为 0，
                    # 但按钮标签会输出到 stdout
                    stdout_out = (rc.stdout or "").strip()
                    if "取消" in stdout_out:
                        return _IDCANCEL
                    return _IDYES
                elif rc.returncode == 1:
                    return _IDNO  # 关闭窗口 / Esc
                return _IDCANCEL
        except Exception as e:
            logger.debug(f"zenity 对话框失败: {e}")
    # 回退：终端 CLI
    return _cli_dialog(title, message, buttons)


def _cli_dialog(title: str, message: str, buttons: str = "ok") -> int:
    """终端 CLI 回退对话框。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}")
    if buttons == "ok":
        _safe_input("按 Enter 继续...", default="")
        return _IDYES
    elif buttons in ("okcancel", "yesno"):
        choice = _safe_input("输入 y/yes 继续，n/no 取消: ", default="n")
        return _IDYES if choice and choice.lower() in ("y", "yes") else _IDNO
    else:
        choice = _safe_input("输入 y/yes=是, n/no=否, c/cancel=取消: ", default="c")
        c = (choice or "c").lower()
        if c in ("y", "yes"):
            return _IDYES
        elif c in ("n", "no"):
            return _IDNO
        return _IDCANCEL


def _show_dialog(title: str, message: str,
                 buttons: str = "ok", owner_hwnd=None) -> int:
    """跨平台对话框：Linux→zenity→CLI, Windows→MessageBox, 其他→CLI。"""
    if IS_LINUX:
        return _show_linux_dialog(title, message, buttons)
    elif IS_WINDOWS:
        flags_map = {
            "ok": _MB_OK | _MB_ICONINFORMATION,
            "okcancel": _MB_OKCANCEL | _MB_ICONQUESTION,
            "yesno": _MB_YESNO | _MB_ICONQUESTION,
            "yesnocancel": _MB_YESNOCANCEL | _MB_ICONQUESTION,
        }
        flags = flags_map.get(buttons, _MB_OK | _MB_ICONINFORMATION)
        return _show_windows_messagebox(title, message, flags, owner_hwnd)
    else:
        return _cli_dialog(title, message, buttons)


def _show_startup_failure(
    title: str,
    message: str,
    startup_splash: _StartupSplash | None = None,
) -> None:
    """Keep startup failures visible in windowed PyInstaller builds."""
    if startup_splash:
        startup_splash.update(100, "启动失败，请查看错误信息")
    logger.error("%s: %s", title, message)
    if startup_splash and startup_splash.enabled:
        _show_dialog(title, message, buttons="ok", owner_hwnd=startup_splash.hwnd)
    elif not getattr(sys, "frozen", False) or _has_interactive_stdin():
        _safe_pause(f"{title}: {message}\n按 Enter 键退出...")
    else:
        # A headless packaged process cannot display a dialog; the log remains
        # the durable diagnostic channel and the process exits deterministically.
        print(f"{title}: {message}", file=sys.stderr)
    if startup_splash:
        startup_splash.close()


def _open_url(url: str):
    """用系统默认浏览器打开 URL（跨平台）。"""
    if IS_LINUX:
        try:
            _sp.run(["xdg-open", url], timeout=10)
            return
        except Exception:
            pass
    else:
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception as e:
        logger.warning(f"无法打开链接 {url}: {e}")


def _find_tailscale_exe() -> str | None:
    """查找 tailscale 可执行文件的完整路径，未安装返回 None（跨平台）。"""
    import shutil
    if IS_WINDOWS:
        candidates = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                        "Tailscale", "tailscale.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)",
                        "C:\\Program Files (x86)"), "Tailscale", "tailscale.exe"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
    # Linux / macOS / Windows PATH fallback
    exe = shutil.which("tailscale")
    if exe:
        return exe
    return None


def _is_tailscale_ip(ip: str | None) -> bool:
    """判断是否是 Tailscale IPv4 CGNAT 或 IPv6 ULA 地址。"""
    if not ip or not isinstance(ip, str):
        return False
    try:
        address = ipaddress.ip_address(ip.strip().split("%", 1)[0])
    except ValueError:
        return False
    return (
        address in ipaddress.ip_network("100.64.0.0/10")
        or address in ipaddress.ip_network("fd7a:115c:a1e0::/48")
    )


def _detect_tailscale_ip_from_interfaces() -> str | None:
    """从网卡接口中检测 Tailscale IP，作为 tailscale CLI 瞬时失败时的兜底。"""
    try:
        import psutil
        import socket
        addrs = psutil.net_if_addrs()
        candidates = []
        for iface, addr_list in addrs.items():
            iface_l = (iface or "").lower()
            for addr in addr_list:
                if addr.family not in (socket.AF_INET, socket.AF_INET6):
                    continue
                ip = getattr(addr, "address", "")
                if not _is_tailscale_ip(ip):
                    continue
                # Tailscale 接口优先，其次接受 100.x 兜底。
                priority = (
                    0 if "tailscale" in iface_l else 1,
                    0 if addr.family == socket.AF_INET else 1,
                )
                candidates.append((priority, ip))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
    except Exception as e:
        logger.debug(f"Tailscale 网卡扫描失败: {e}", exc_info=True)
    return None


def _check_tailscale_status() -> dict:
    """
    检查本机 Tailscale 状态。

    检测采用多来源兜底：
      1. tailscale status --json（短重试，获取 hostname/self 信息）
      2. tailscale ip -4（CLI JSON 失败时仍可返回 IP）
      3. 网卡接口 100.x / Tailscale IP 扫描

    只要检测到可用 Tailscale IP，就认为可以继续启动，避免打包版启动时
    因 tailscale 服务刚启动、CLI status 暂时非 0 等瞬时状态误弹“未连接”。
    """
    result = {
        "installed": False,
        "running": False,
        "logged_in": False,
        "tailscale_ip": None,
        "hostname": None,
        "source": "none",
        "error_detail": "",
    }
    exe = _find_tailscale_exe()
    if not exe:
        result["error_detail"] = "tailscale.exe not found"
        return result
    result["installed"] = True

    errors = []

    # 1) status --json：启动路径只做一次短探测，失败立即走 IP/网卡兜底。
    try:
        r = _sp.run(
            [exe, "status", "--json"],
            capture_output=True, text=True, timeout=2,
            encoding="utf-8", errors="replace",
            creationflags=_WINDOWS_NO_WINDOW,
        )
        if r.returncode == 0 and r.stdout.strip():
            import json
            data = json.loads(r.stdout)
            self_node = data.get("Self", {}) or {}
            ips = self_node.get("TailscaleIPs", []) or []
            ts_ip = next((ip for ip in ips if _is_tailscale_ip(ip)), ips[0] if ips else None)
            result["running"] = True
            if self_node:
                result["logged_in"] = True
                result["hostname"] = self_node.get("HostName")
            if ts_ip:
                result["tailscale_ip"] = ts_ip
                result["source"] = "status_json"
                return result
            errors.append("status json has no Tailscale IP")
        else:
            stderr = (r.stderr or "").strip()
            errors.append(f"status --json rc={r.returncode}: {stderr[:160]}")
    except Exception as e:
        errors.append(f"status --json: {e}")

    # 2) tailscale ip：status JSON 不可用时分别尝试 IPv4/IPv6。
    for family in ("-4", "-6"):
        try:
            r = _sp.run(
                [exe, "ip", family],
                capture_output=True, text=True, timeout=2,
                encoding="utf-8", errors="replace",
                creationflags=_WINDOWS_NO_WINDOW,
            )
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.splitlines():
                    ip = line.strip()
                    if _is_tailscale_ip(ip):
                        result.update({
                            "running": True,
                            "logged_in": True,
                            "tailscale_ip": ip,
                            "source": "tailscale_ip",
                        })
                        logger.info(
                            "Tailscale status JSON 不可用，已通过 `tailscale ip %s` 确认: %s",
                            family,
                            ip,
                        )
                        return result
                errors.append(f"tailscale ip {family} returned no trusted IP")
            else:
                stderr = (r.stderr or "").strip()
                errors.append(f"tailscale ip {family} rc={r.returncode}: {stderr[:160]}")
        except Exception as e:
            errors.append(f"tailscale ip {family}: {e}")

    # 3) 网卡扫描：最终兜底。
    ip = _detect_tailscale_ip_from_interfaces()
    if ip:
        result.update({
            "running": True,
            "logged_in": True,
            "tailscale_ip": ip,
            "source": "interface",
        })
        logger.info(f"Tailscale CLI 暂不可用，已通过网卡/IP 检测确认可用: {ip}")
        return result

    result["error_detail"] = " | ".join(errors[-4:])
    if result["error_detail"]:
        logger.warning(f"Tailscale 状态检测未获取到 IP: {result['error_detail']}")
    return result


def _prompt_tailscale_setup(status: dict, owner_hwnd=None) -> bool:
    """
    首次启动时引导用户安装/加入 Tailscale 组网。

    打包版为 console=False，没有 stdin；因此优先使用 Windows 消息框，避免
    input() 在 windowed 进程中触发 "lost sys.stdin" 崩溃。

    Args:
        status: _check_tailscale_status() 的返回值

    Returns:
        True = 用户确认继续（已加入或选择暂时跳过时也返回 True）
    """
    installed = status["installed"]
    logged_in = status["logged_in"]
    ts_ip = status.get("tailscale_ip")

    # 已安装且已登录且有 IP → 静默通过
    if installed and logged_in and ts_ip:
        return True

    if not installed:
        problem = "未检测到 Tailscale。"
        steps = (
            "1. 下载并安装 Tailscale\n"
            f"   {TAILSCALE_DOWNLOAD_URL}\n\n"
            "2. 安装完成后，通过邀请链接加入组网\n"
            f"   {TAILSCALE_INVITE_URL}\n\n"
            "3. 登录后，确认系统托盘 Tailscale 图标显示 Connected。"
        )
    elif not logged_in:
        problem = "Tailscale 已安装，但未登录/未加入组网。"
        steps = (
            "请通过邀请链接完成加入：\n"
            f"{TAILSCALE_INVITE_URL}\n\n"
            "加入后确认系统托盘 Tailscale 图标显示 Connected。"
        )
    else:
        problem = "Tailscale 已登录，但未获取到 100.x IP。"
        steps = "请检查 Tailscale 网络状态，确认已 Connected。"

    message = (
        "QLH 分布式推理建议使用 Tailscale 实现跨子网互联。\n\n"
        f"当前状态：{problem}\n\n"
        f"{steps}\n\n"
        "选择“是”：打开相关链接，稍后请重新启动程序。\n"
        "选择“否”：本次先跳过检查，继续启动单机/本机 Web 服务。\n"
        "选择“取消”：退出程序。"
    )

    # 无控制台的安装包路径：使用系统对话框交互。
    if not _has_interactive_stdin():
        result = _show_dialog(
            "Tailscale 组网检查",
            message,
            "yesnocancel",
            owner_hwnd=owner_hwnd,
        )
        if result == _IDYES:
            if not installed:
                _open_url(TAILSCALE_DOWNLOAD_URL)
            _open_url(TAILSCALE_INVITE_URL)
            return False
        if result == _IDNO:
            logger.warning("用户选择暂时跳过 Tailscale 检查，继续启动。")
            return True
        return False

    # ---- 控制台开发模式：显示完整引导界面 ----
    print()
    print("=" * 60)
    print("  🔗 Tailscale 组网检查")
    print("=" * 60)
    print()
    print("  QLH 分布式推理需要节点间直接通信。")
    print("  由于校园网不同子网之间相互隔离，")
    print("  系统采用 Tailscale 虚拟组网实现跨子网互联。")
    print()
    print(f"  ⚠️  {problem}")
    print()
    for line in steps.splitlines():
        print(f"  {line}")
    print()
    print("─" * 60)
    print()
    print("  加入组网后，请在下面输入 yes 继续。")
    print("  输入 skip 可本次跳过检查，仅用于单机/本机 Web 服务。")
    print("  输入 no 将退出程序。")
    print()

    while True:
        choice_raw = _safe_input("  >>> 是否已加入 Tailscale 组网？(yes/skip/no): ", default="no")
        choice = (choice_raw or "no").strip().lower()

        if choice in ("yes", "y"):
            # 再次检查状态，确认用户真的加入了
            new_status = _check_tailscale_status()
            if new_status.get("logged_in") and new_status.get("tailscale_ip"):
                # 写入跳过标志，下次启动不再提示
                try:
                    os.makedirs(_TAILSCALE_FLAG_DIR, exist_ok=True)
                    with open(_TAILSCALE_FLAG_FILE, "w") as f:
                        f.write(f"joined=1\n")
                        f.write(f"tailscale_ip={new_status['tailscale_ip']}\n")
                except Exception:
                    pass
                logger.info(f"Tailscale 组网已就绪: {new_status['tailscale_ip']}")
                return True
            else:
                print()
                print("  ⚠️  仍检测不到 Tailscale 连接。请确认：")
                print("     1. Tailscale 已安装并运行")
                print("     2. 已通过邀请链接加入组网")
                print("     3. 系统托盘图标显示 Connected")
                print()
        elif choice in ("skip", "s"):
            logger.warning("用户选择暂时跳过 Tailscale 检查，继续启动。")
            return True
        elif choice in ("no", "n"):
            print()
            print("  已取消。请加入 Tailscale 组网后重新启动程序。")
            return False
        else:
            print("  请输入 yes、skip 或 no。")


def _check_tailscale_requirement(owner_hwnd=None) -> bool:
    """
    检查 Tailscale 组网要求。

    逻辑:
    - 若标记文件存在 → 静默快速检查（已登录 → 跳过；未登录 → 提示）
    - 若标记文件不存在 → 首次启动，显示完整引导流程

    Returns:
        True = 可以继续启动
        False = 应退出程序
    """
    # 已标记为加入 → 快速检查
    if os.path.isfile(_TAILSCALE_FLAG_FILE):
        status = _check_tailscale_status()
        if status.get("logged_in") and status.get("tailscale_ip"):
            logger.info(f"Tailscale 在线: {status['tailscale_ip']}")
            return True
        else:
            # 标记存在但 Tailscale 未运行（可能未开机自启）
            logger.warning("Tailscale 已标记但当前未连接，提示重新连接")
            print()
            print("  ⚠️  Tailscale 当前未连接。")
            print("  请确保 Tailscale 正在运行且已连接到组网。")
            print(f"  如有问题，请重新访问邀请链接: {TAILSCALE_INVITE_URL}")
            print()
            return _prompt_tailscale_setup(status, owner_hwnd=owner_hwnd)

    # 首次启动 → 完整引导
    status = _check_tailscale_status()
    return _prompt_tailscale_setup(status, owner_hwnd=owner_hwnd)


def _torch_available() -> bool:
    """PyTorch 是否**装得上**（不导入它，避免拖起整棵依赖树）。

    与 `_detect_cuda()` 的区别：那个回答"有 GPU 吗"，这个回答"有 PyTorch 吗"。
    边缘档（`qlh-edge.spec`）无 torch 但可能有别的加速器；而 CPU 档可能有 torch
    却没 GPU —— 决定"能不能导入 `model_module`"的是后者。
    """
    import importlib.util

    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _detect_cuda() -> bool:
    """静默检测 CUDA 可用性。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_engine_preference() -> str:
    """
    检测推荐的推理引擎。

    Returns:
        "llama_cpp" 或 "pytorch"
    """
    from config import INFERENCE_ENGINE

    if INFERENCE_ENGINE == "llama_cpp":
        return "llama_cpp"
    if INFERENCE_ENGINE == "pytorch":
        return "pytorch"

    # "auto" 模式
    if _detect_cuda():
        return "pytorch"
    return "llama_cpp"


def _check_model_assets_for_startup() -> bool:
    """Check model assets without presenting a blocking download dialog."""
    from model_downloader import ensure_model_or_warn

    return bool(ensure_model_or_warn())


def _has_webview() -> bool:
    """检测 pywebview 是否可用（仅 Windows 支持原生窗口）。"""
    if not IS_WINDOWS:
        return False
    try:
        import webview  # noqa: F401
        return True
    except Exception:
        return False


def _run_ui(url: str, title: str, startup_splash: _StartupSplash | None = None):
    """
    启动用户界面（跨平台）。

    Linux: 使用 xdg-open 打开系统浏览器。
    Windows: 优先使用 pywebview 原生窗口，不可用时回退到外部浏览器。
    """
    if IS_LINUX or not _has_webview():
        if startup_splash:
            startup_splash.close()
        _launch_browser(url)
    else:
        _run_pywebview(url, title, startup_splash=startup_splash)


def _run_pywebview(url: str, title: str,
                   startup_splash: _StartupSplash | None = None):
    """
    Windows pywebview 原生窗口。

    如果 pywebview 不可用或启动失败，回退到外部浏览器。
    """
    try:
        import webview
    except Exception:
        logger.warning("pywebview 未安装，回退到外部浏览器")
        if startup_splash:
            startup_splash.close()
        _launch_browser(url)
        return

    try:
        window = webview.create_window(
            title=title,
            url=url,
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            confirm_close=False,
        )

        splash_closed = threading.Event()

        def close_startup_splash(*_args):
            if startup_splash and not splash_closed.is_set():
                splash_closed.set()
                startup_splash.close()

        def on_started():
            # The GUI loop being ready does not mean the page has rendered. Keep
            # the startup window until pywebview reports a completed page load.
            timer = threading.Timer(20.0, close_startup_splash)
            timer.daemon = True
            timer.start()

        if hasattr(window.events, "loaded"):
            window.events.loaded += close_startup_splash
        else:
            logger.warning("pywebview 不提供 loaded 事件，使用启动页超时兜底")

        def on_closed():
            close_startup_splash()
            logger.info("窗口已关闭，程序退出。")

        # Override the earlier handler so closing a window during navigation also
        # tears down the splash immediately.
        window.events.closed += on_closed
        webview.start(on_started, gui='edgechromium', debug=False)
    except Exception as e:
        logger.warning(f"pywebview 启动失败 ({e})，回退到外部浏览器")
        if startup_splash:
            startup_splash.close()
        _launch_browser(url)


def _configure_console_appearance() -> None:
    """把 `AllocConsole()` 新建的控制台调成与 Windows Terminal 接近的外观。

    `AllocConsole()` 给的是**经典 conhost**：默认点阵字体、16 色配色，**且不解析 ANSI
    转义序列**。Textual 整个界面都靠 VT/ANSI 序列绘制 ⇒ 不启用 VT 就会看到颜色错乱、
    框线与色块画不出来；默认点阵字体也画不清那些方块字符。实测：双击 exe 起的 TUI
    与在 Windows Terminal 里跑的同一个 TUI，外观明显不同，根源就是这三项。

    全部失败都不致命（拿不到控制台句柄时静默跳过）—— TUI 仍能跑，只是外观退化。
    """
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        ENABLE_PROCESSED_OUTPUT = 0x0001
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

        # ① 启用 VT 处理 —— Textual 的 ANSI 序列靠它生效（最关键的一项）
        mode = wintypes.DWORD()
        if handle and kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(
                handle,
                mode.value | ENABLE_PROCESSED_OUTPUT
                | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
            )

        # ② 输入输出都用 UTF-8（界面里有 CJK 与制表符）
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)

        # ③ 字体换等宽 Consolas（默认点阵字体画不清框线与块元素）
        class _COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class _CONSOLE_FONT_INFOEX(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.ULONG),
                ("nFont", wintypes.DWORD),
                ("dwFontSize", _COORD),
                ("FontFamily", wintypes.UINT),
                ("FontWeight", wintypes.UINT),
                ("FaceName", wintypes.WCHAR * 32),
            ]

        info = _CONSOLE_FONT_INFOEX()
        info.cbSize = ctypes.sizeof(_CONSOLE_FONT_INFOEX)
        info.dwFontSize = _COORD(0, 18)
        info.FontFamily = 54          # FF_MODERN | FIXED_PITCH
        info.FontWeight = 400
        info.FaceName = "Consolas"
        kernel32.SetCurrentConsoleFontEx(
            handle, False, ctypes.byref(info),
        )
    except Exception:  # noqa: BLE001 - 外观优化失败不该阻断 TUI
        logger.debug("控制台外观设置失败（不影响 TUI 运行）", exc_info=True)


def _ensure_tui_console() -> bool:
    """Attach a console for a windowed Windows build before entering TUI."""
    if _has_interactive_stdin():
        # ★ 2026-10-03：命中这条说明控制台**已经存在**（`console=True` 的 exe 自带，
        #   或用户从终端启动）。外观仍要调 —— 自带的是经典 conhost，默认不解析 ANSI
        #   转义 ⇒ Textual 的颜色与框线会花掉（实测）。从 Windows Terminal 启动时
        #   这几项已是启用状态，重复设置无副作用。
        _configure_console_appearance()
        return True
    if not IS_WINDOWS or not getattr(sys, "frozen", False):
        return False
    try:
        import ctypes

        if not ctypes.windll.kernel32.AllocConsole():
            return False
        sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
        sys.stdout = open(
            "CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1,
        )
        sys.stderr = open(
            "CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1,
        )
        ctypes.windll.kernel32.SetConsoleTitleW("QLH · BJTU TUI 管理")
        # ★ 新建的 conhost 默认不解析 ANSI、字体是点阵 ⇒ 不调这个 TUI 颜色/框线全乱。
        _configure_console_appearance()
        return True
    except Exception:
        logger.exception("无法为 TUI 创建控制台")
        return False


def _run_tui(
    api_port: int, startup_splash: _StartupSplash | None = None,
) -> int:
    """Enter the existing management TUI after the shared backend is ready."""
    if startup_splash:
        startup_splash.close()
    if not _ensure_tui_console():
        _show_dialog(
            "TUI 无法启动",
            "当前环境没有可交互终端。请在终端运行 bjtu tui，"
            "或改用普通界面。",
            buttons="ok",
        )
        return 1
    # ★ 2026-10-03：入口从 `tui_admin` 改为 `tui_textual` —— commit `656935e6`
    #   「归档旧标准库 TUI」之后 `src/tui_admin.py` 已不存在，`tui_textual.py` 才是
    #   带 `main()`/`__main__` 的 Textual 管理界面（也是 `doc-style.md` 引为 TUI
    #   页面/命令数事实源的那份）。
    from tui_textual import main as tui_main

    print("QLH 服务已就绪，正在进入 BJTU TUI 管理界面……")
    return int(tui_main(["--port", str(api_port)]) or 0)


def _run_selected_interface(
    mode: str, url: str, title: str, startup_splash: _StartupSplash | None = None,
) -> int:
    if mode == "tui":
        from urllib.parse import urlparse

        api_port = urlparse(url).port or 8000
        return _run_tui(api_port, startup_splash=startup_splash)
    _run_ui(url=url, title=title, startup_splash=startup_splash)
    return 0


def _launch_browser(url: str):
    """跨平台打开系统浏览器。Linux 用 xdg-open，Windows 用 startfile，通用回退 webbrowser。"""
    import webbrowser

    methods = []
    if IS_LINUX:
        methods.append(lambda: _sp.run(["xdg-open", url], timeout=10))
    else:
        methods.append(lambda: os.startfile(url))
        methods.append(lambda: _sp.run(["cmd", "/c", "start", url],
                                       capture_output=True, timeout=5,
                                       creationflags=_WINDOWS_NO_WINDOW))
    methods.append(lambda: webbrowser.open(url))

    for method in methods:
        try:
            method()
            logger.info("浏览器已打开: " + url)
            return
        except Exception:
            continue

    logger.info("请手动打开浏览器访问: " + url)
    print(f"\n{'='*60}")
    print(f"  请手动打开浏览器访问:")
    print(f"  >>> {url} <<<")
    print(f"{'='*60}\n")
    _safe_input("按 Enter 键退出...", default="")


def _is_existing_qlh_instance(port: int) -> bool:
    """Return True when the listening local API is an existing QLH instance."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/cluster/my-role",
            timeout=0.8,
        ) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
        return payload.get("run_mode") in {"single", "distributed"} and bool(
            payload.get("node_id")
        )
    except Exception:
        return False


def _kill_port_8000(port: int = 8000) -> str:
    """
    检查 API 端口并区分已有 QLH 实例与其他占用者。

    Returns: ``free`` | ``qlh`` | ``occupied``
    """
    import socket

    def _port_in_use() -> bool:
        """检测 API 端口是否被占用。"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            s.connect(("127.0.0.1", port))
            return True
        except (OSError, ConnectionRefusedError):
            return False
        finally:
            s.close()

    if not _port_in_use():
        return "free"

    if _is_existing_qlh_instance(port):
        logger.info("检测到已有 QLH 实例: http://127.0.0.1:%s", port)
        return "qlh"
    logger.error("端口 %s 已被其他程序占用", port)
    return "occupied"


def _verify_pytorch_tokenizer_runtime() -> str:
    """Load the local PyTorch tokenizer so packaging checks cover dynamic imports."""
    from config import MODEL_PATH
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    token_ids = tokenizer.encode("QLH tokenizer runtime check")
    if not token_ids:
        raise RuntimeError("Qwen tokenizer returned no token IDs")
    tokenizer_name = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
    logger.info("PyTorch tokenizer runtime check passed: %s", tokenizer_name)
    return tokenizer_name


def _run_keephead_worker(argv: list[str]) -> int:
    """把包内的 ``llama_keep_head_worker.py`` 当作 ``__main__`` 执行。

    仅对**冻结包**有意义：此时 ``sys.executable`` 是本 exe 自己、不是 Python 解释器，
    ``llama_keep_head._init_isolated_worker()`` 不能再写
    ``[sys.executable, worker.py, ...]`` —— 那只会让本 exe 重跑 launcher 入口，
    worker 的 stdin/stdout 协议永不建立。2026-10-06 用 Edge 包当 layer worker 实测踩到：
    子进程在（37MB）、``keephead-worker.log`` 为 0 字节、master 等满 60s 超时。
    这里早期分派，把同一份 worker 脚本直接跑起来（包内已含它所需的全部模块）。
    """
    import runpy

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "llama_keep_head_worker.py")
    if not os.path.isfile(worker):
        print(f"[keephead-worker] missing: {worker}", file=sys.stderr)
        return 2
    sys.argv = [worker, *argv]
    runpy.run_path(worker, run_name="__main__")
    return 0


def _application_ready(response):
    if response.status != 200:
        return False
    try:
        return json.loads(response.read()).get("ready") is True
    except (ValueError, AttributeError):
        return False


def _shutdown_owned_backend(thread, *, timeout=15.0):
    from api_server import stop_api_servers
    stop_api_servers()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise RuntimeError("application_shutdown_cleanup_pending")


def main():
    if "--keephead-worker" in sys.argv:
        return _run_keephead_worker(sys.argv[sys.argv.index("--keephead-worker") + 1:])
    _redirect_frozen_logs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    """启动器主入口（跨平台）。

    CLI 参数:
      --ui          直接打开普通 Web/原生界面
      --tui         启动后端后进入终端管理界面
      --headless    跳过浏览器/窗口，仅后台运行 API 服务器（适合 systemd / 无头部署）
      --check-only  仅检查环境，打印状态后退出（CI/测试用）
      --keephead-worker  内部使用：以 __main__ 跑包内的 keep-head 隔离 worker
    """
    headless = "--headless" in sys.argv
    check_only = "--check-only" in sys.argv
    try:
        launch_mode = _requested_launch_mode(sys.argv[1:])
    except ValueError as exc:
        _show_dialog("启动参数冲突", str(exc), buttons="ok")
        return
    if headless or check_only:
        launch_mode = "headless"
    startup_splash = _StartupSplash(enabled=False).start()
    import atexit
    atexit.register(startup_splash.close)
    startup_splash.update(5, "正在初始化运行环境...")
    from config import API_PORT

    # ---- 单实例检查（环境自检不需要占用 API 端口） ----
    if not check_only:
        startup_splash.update(10, "正在检查应用运行状态...")
        port_status = _kill_port_8000(API_PORT)
        if port_status == "qlh":
            if headless:
                logger.info("已有 QLH 服务正在运行，无头启动直接复用")
            else:
                startup_splash.update(100, "正在打开应用窗口...")
                _run_selected_interface(
                    launch_mode,
                    url=f"http://localhost:{API_PORT}",
                    title="轻量化大模型分布式边缘推理系统",
                    startup_splash=startup_splash,
                )
            return
        if port_status == "occupied":
            startup_splash.close()
            _show_dialog(
                "端口被占用",
                f"端口 {API_PORT} 已被其他程序占用，QLH 无法启动。\n"
                "请关闭占用程序后重试。",
                buttons="ok",
            )
            return

    # ---- 确定引擎 ----
    startup_splash.update(18, "正在检测推理引擎...")
    engine = _detect_engine_preference()
    has_cuda = _detect_cuda()

    print("=" * 60)
    print("  轻量化大模型分布式边缘推理优化系统")
    if has_cuda:
        print("  独显版本 — PyTorch + bitsandbytes INT4")
    else:
        print("  集显版本 (CPU-only) — llama.cpp + GGUF Q4_K_M")
    print("  北京交通大学 · 大学生创新创业训练计划")
    print(f"  平台: {'Linux' if IS_LINUX else 'Windows'}")
    print("=" * 60)

    if engine == "llama_cpp":
        print("  🚀 推理引擎: llama.cpp (CPU/集显 优化)")
        print("     模型: GGUF Q4_K_M (~1.16 GB)")
        print("     预计速度: 10-15 tok/s (4核 CPU)")
    else:
        print("  🚀 推理引擎: PyTorch + bitsandbytes")
        print("     模型: Safetensors (~3.6 GB)")
        print("     量化: INT4 (~1.75 GB 显存)")
    if headless:
        print("  🤖 无头模式: API 服务器 + 无浏览器")
    print()

    # ---- 第 0 步：Tailscale 组网检查 ----
    startup_splash.update(28, "正在检查集群网络...")
    if not _check_tailscale_requirement(owner_hwnd=startup_splash.hwnd):
        startup_splash.close()
        print()
        print("按 Enter 键退出...")
        _safe_input(default="")
        sys.exit(1)
    print()

    # ---- 第 1 步：检查模型文件 ----
    # ★ 2026-10-03：先确保 `models/` 存在。`config._APP_ROOT` 在打包形态下是
    #   **exe 所在目录**（`config.py` 的注释即"models/ 与 exe 同级"），而
    #   `model_downloader.gguf_model_exists()` 第一步就 `os.path.isdir(GGUF_DIR)` ——
    #   目录不存在时直接 False ⇒ **任何**模型都被判"未落盘"，用户把模型放进去也找不到。
    #   只建 `models/` 本身：**不再**建 `models/qwen-1_8b-chat/` 之类的模型子目录 ——
    #   那是 Qwen-1.8B 时代的遗留，该模型已退役，建出来只会往用户的模型目录里掺空壳。
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(sys.executable)), "models",
    )
    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError:
        logger.debug("创建模型目录失败：%s", models_dir, exc_info=True)

    startup_splash.update(42, "正在检查本地模型...")
    from model_downloader import (
        gguf_model_exists,
        safetensors_model_exists,
    )

    model_ready = _check_model_assets_for_startup()
    if check_only and not model_ready:
        logger.error("No local model is installed; --check-only cannot pass")
        print("[ERROR] No local model is installed; --check-only requires a model asset.")
        sys.exit(1)
    if not model_ready:
        logger.warning("No local model is installed; continuing to the model download workspace")
        print(f"No local model is installed. Put a .gguf file under: {models_dir}")
    if False:  # legacy blocking-model branch retained below for source compatibility
        logger.warning("No local model is installed; continuing to the model download workspace")
        print()
        print("模型文件未就绪，程序将退出。")
        if engine == "llama_cpp":
            print("请下载 GGUF 格式模型后重新启动。")
            print("推荐: Qwen-1_8B-Chat-Q4_K_M.gguf (~1.16 GB)")
            print("下载: https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf")
        else:
            print("请下载 Safetensors 格式模型后重新启动。")
            print("下载: https://huggingface.co/Qwen/Qwen-1.8B-Chat")
        print()
        print("按 Enter 键退出...")
        # A missing model is a recoverable application state, not a launcher failure.

    # ---- 报告检测结果 ----
    has_gguf = gguf_model_exists()
    has_safetensors = safetensors_model_exists()

    if has_gguf:
        logger.info("✅ GGUF 模型就绪 (llama.cpp)")
    if has_safetensors:
        logger.info("✅ Safetensors 模型就绪 (PyTorch)")

    # ---- 确认引擎选择 ----
    startup_splash.update(58, "正在准备模型运行环境...")
    # ★ 2026-10-03：`model_module` 是 **PyTorch 引擎**模块 —— 它顶层无条件
    #   `import torch` / `import transformers`。边缘档（`qlh-edge.spec`）**刻意不打
    #   torch**（推理走 llama.cpp/GGUF）⇒ 这里不能无条件导入它，否则 launcher 在
    #   导入期就 `ModuleNotFoundError: No module named 'torch'` 直接崩（实测）。
    #   判据用「torch 是否装得上」而不是「CUDA 是否可用」：边缘档无 torch，
    #   CPU 档可能装了 torch 但没有 GPU —— 前者必须跳过导入，后者仍需 ModelManager。
    if _torch_available():
        from model_module import ModelManager

        actual_engine = ModelManager.select_engine()
    else:
        actual_engine = "llama_cpp"
        logger.info("未检测到 PyTorch（边缘档），推理引擎固定为 llama_cpp（GGUF）")
    logger.info(f"推理引擎: {actual_engine}")

    if check_only:
        if has_safetensors:
            try:
                tokenizer_name = _verify_pytorch_tokenizer_runtime()
            except Exception as e:
                logger.exception("PyTorch tokenizer runtime check failed")
                print(f"[ERROR] PyTorch tokenizer runtime check failed: {e}")
                sys.exit(1)
            print(f"   Tokenizer: {tokenizer_name}")
        print()
        print("✅ 环境检查通过。模型就绪，引擎已选择。")
        print(f"   引擎: {actual_engine}")
        print(f"   平台: {'Linux' if IS_LINUX else 'Windows'}")
        print(f"   CUDA: {'可用' if has_cuda else '不可用'}")
        return

    # ---- 第 2 步：后台启动 API 服务器 ----
    startup_splash.update(70, "正在启动本地服务...")
    print("正在加载 API 服务...")
    _server_error = []

    def run_server():
        """在后台线程中启动 uvicorn（IPv4/IPv6 双栈）。"""
        try:
            from api_server import run_api_servers
            run_api_servers("0.0.0.0", API_PORT)
        except Exception as e:
            _server_error.append(str(e))
            import traceback
            _server_error.append(traceback.format_exc())
            print(f"\n[ERROR] 服务器启动失败: {e}\n", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    server_thread = threading.Thread(target=run_server, daemon=True, name="uvicorn")
    server_thread.start()

    try:
        # 等待服务器就绪。首次导入 torch/transformers 时可能超过 15 秒；
        # 启动页持续显示阶段，超时后给出可见错误而不是静默退出。
        print("等待 API 服务器就绪...")
        import urllib.request
        server_ready = False
        startup_timeout = _startup_timeout_seconds()
        poll_interval = 0.1
        poll_count = max(1, int(startup_timeout / poll_interval))
        for i in range(poll_count):
            time.sleep(0.1)
            if i % 5 == 0:
                startup_splash.update(
                    72 + min(20, int(i / poll_count * 20)),
                    "正在等待本地服务就绪...",
                )
            if _server_error:
                _show_startup_failure(
                    "QLH 服务启动失败",
                    f"{_server_error[0]}\n\n详细日志目录：{_launcher_log_hint()}",
                    startup_splash,
                )
                sys.exit(1)
            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{API_PORT}/api/ready", timeout=0.5,
                )
                if _application_ready(resp):
                    server_ready = True
                    break
            except Exception:
                if i % 20 == 19:
                    print(f"  等待中... ({int(i * 0.1 + 1)}s)")
        if not server_ready:
            if _server_error:
                detail = _server_error[0]
            else:
                detail = (
                    f"本地服务在 {startup_timeout:.0f} 秒内没有响应。"
                    "\n可能仍在加载模型或数据库，请查看日志后重试。"
                )
            _show_startup_failure(
                "QLH 服务启动超时",
                f"{detail}\n\n详细日志目录：{_launcher_log_hint()}",
                startup_splash,
            )
            sys.exit(1)

        print(f"API 服务器已就绪: http://localhost:{API_PORT}")
        startup_splash.update(96, "本地服务已就绪，正在加载界面...")

        # ---- 第 3 步：启动用户界面 ----
        if headless:
            print()
            print("🤖 无头模式: API 服务器运行中，按 Ctrl+C 退出。")
            print(f"   访问 http://localhost:{API_PORT} 打开 Web 界面。")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n正在退出...")
        else:
            print()
            print("启动用户界面...")
            startup_splash.update(100, "正在打开应用窗口...")
            _run_selected_interface(
                launch_mode,
                url=f"http://localhost:{API_PORT}",
                title="轻量化大模型分布式边缘推理系统",
                startup_splash=startup_splash,
            )

    finally:
        startup_splash.close()
        _shutdown_owned_backend(server_thread)
    print("程序已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
