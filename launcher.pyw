"""
Stock Trading AI Platform Launcher
Starts the Streamlit dashboard with no terminal window.
Includes health monitoring and auto-restart.
"""
import subprocess
import threading
import time
import os
import sys
import tkinter as tk
from tkinter import font as tkfont

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable.replace("pythonw.exe", "python.exe")
LOG_FILE = os.path.join(BASE_DIR, "dashboard.log")
PORT = 8501

# Process
server_proc = None
shutting_down = False


def start_server():
    """Start the Streamlit dashboard server."""
    global server_proc
    log_fh = open(LOG_FILE, "a", encoding="utf-8")

    env = os.environ.copy()
    # Suppress Streamlit email prompt and telemetry
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"

    server_proc = subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run", "src/dashboard/app.py",
         "--server.port", str(PORT),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=BASE_DIR,
        stdout=log_fh,
        stderr=log_fh,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def health_check():
    """Monitor server process, auto-restart if it dies."""
    global server_proc
    while not shutting_down:
        time.sleep(10)
        if shutting_down:
            break
        if server_proc and server_proc.poll() is not None:
            update_status("Dashboard crashed — restarting...", "#ff6b6b")
            start_server()
            time.sleep(3)
            if server_proc and server_proc.poll() is None:
                update_status("Dashboard running", "#4ade80")


def update_status(text, color):
    """Thread-safe status update."""
    try:
        root.after(0, lambda: [status_var.set(text), status_label.config(fg=color)])
    except Exception:
        pass


def kill_tree(proc):
    """Kill a process and its entire child tree."""
    if proc is None:
        return
    try:
        subprocess.call(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def kill_port(port):
    """Kill every process listening on the given port."""
    try:
        result = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                f"Get-NetTCPConnection -LocalPort {port} -State Listen "
                f"-ErrorAction SilentlyContinue | "
                f"Select-Object -ExpandProperty OwningProcess"
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in result.splitlines():
            pid = line.strip()
            if pid.isdigit():
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", pid],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    except Exception:
        pass


def stop_all():
    """Kill server and any orphan processes on the port."""
    global shutting_down, server_proc
    shutting_down = True
    kill_tree(server_proc)
    kill_port(PORT)
    server_proc = None


def do_stop():
    """Stop button handler."""
    stop_all()
    stop_btn.config(text="Restart", fg="#4ade80", command=do_restart)
    update_status("Dashboard stopped. Click Restart to start again.", "#888")


def do_restart():
    """Restart button handler."""
    global shutting_down
    shutting_down = False
    stop_btn.config(text="Stop", fg="#ff6b6b", command=do_stop)
    update_status("Starting dashboard...", "#888")
    t = threading.Thread(target=boot, daemon=True)
    t.start()


def do_close():
    """Close the window and kill everything."""
    stop_all()
    root.destroy()


def open_dashboard():
    os.startfile(f"http://localhost:{PORT}")


def open_log():
    if os.path.exists(LOG_FILE):
        os.startfile(LOG_FILE)


# ---- GUI ----
root = tk.Tk()
root.title("Stock Trading AI")
root.configure(bg="#0d1117")
root.resizable(False, False)

win_w, win_h = 480, 300
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
x = (screen_w - win_w) // 2
y = (screen_h - win_h) // 2
root.geometry(f"{win_w}x{win_h}+{x}+{y}")
root.protocol("WM_DELETE_WINDOW", do_close)

try:
    root.iconbitmap(default="")
except Exception:
    pass

# Fonts
title_font = tkfont.Font(family="Segoe UI", size=18, weight="bold")
label_font = tkfont.Font(family="Segoe UI", size=10)
url_font = tkfont.Font(family="Consolas", size=11)
btn_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
status_font = tkfont.Font(family="Segoe UI", size=9)

# Colors
BG = "#0d1117"
FG = "#e0e0e0"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
BTN_BG = "#161b22"
BTN_ACTIVE = "#21262d"

tk.Label(root, text="Stock Trading AI", font=title_font,
         fg=GREEN, bg=BG).pack(pady=(20, 5))

tk.Label(root, text="Dashboard running at:", font=label_font,
         fg=FG, bg=BG).pack(pady=(5, 10))

url_var = tk.StringVar(value=f"http://localhost:{PORT}")
url_entry = tk.Entry(root, textvariable=url_var, font=url_font,
                     fg=ACCENT, bg="#161b22", bd=0,
                     readonlybackground="#161b22",
                     insertbackground=ACCENT,
                     justify="center", state="readonly",
                     width=40)
url_entry.pack(padx=20, ipady=8)

btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(pady=15)

open_btn = tk.Button(btn_frame, text="Open Dashboard", font=btn_font,
                     fg=FG, bg=BTN_BG, activebackground=BTN_ACTIVE,
                     activeforeground=FG, bd=0, padx=20, pady=6,
                     cursor="hand2", command=open_dashboard)
open_btn.pack(side="left", padx=8)

stop_btn = tk.Button(btn_frame, text="Stop", font=btn_font,
                     fg="#ff6b6b", bg=BTN_BG, activebackground=BTN_ACTIVE,
                     activeforeground="#ff6b6b", bd=0, padx=20, pady=6,
                     cursor="hand2", command=do_stop)
stop_btn.pack(side="left", padx=8)

log_btn = tk.Button(btn_frame, text="View Log", font=btn_font,
                    fg=FG, bg=BTN_BG, activebackground=BTN_ACTIVE,
                    activeforeground=FG, bd=0, padx=20, pady=6,
                    cursor="hand2", command=open_log)
log_btn.pack(side="left", padx=8)

status_var = tk.StringVar(value="Starting dashboard...")
status_label = tk.Label(root, textvariable=status_var, font=status_font,
                        fg="#888", bg=BG)
status_label.pack(pady=(5, 0))

tk.Label(root, text="Stop kills the server. Restart brings it back. x closes.",
         font=status_font, fg="#555", bg=BG, justify="center").pack(pady=(10, 0))


def boot():
    update_status("Starting dashboard...", "#888")
    kill_port(PORT)  # Kill any orphan from previous session
    time.sleep(1)
    start_server()
    # Wait for server to be ready
    time.sleep(4)
    if server_proc and server_proc.poll() is None:
        update_status("Dashboard running", GREEN)
        root.after(0, lambda: open_dashboard())
    else:
        update_status("Failed to start — check log", "#ff6b6b")


boot_thread = threading.Thread(target=boot, daemon=True)
boot_thread.start()

health_thread = threading.Thread(target=health_check, daemon=True)
health_thread.start()

root.mainloop()

# Final cleanup
stop_all()
