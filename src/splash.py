"""
启动画面 — tkinter 实现，在重型模块加载时提供即时视觉反馈
"""

import tkinter as tk


class SplashScreen:
    def __init__(self, title: str, subtitle: str = ""):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.configure(bg="#1a1a2e")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w, win_h = 420, 140
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        outer = tk.Frame(self.root, bg="#1a1a2e", padx=3, pady=3)
        outer.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(outer, bg="#16213e", highlightbackground="#0f3460",
                         highlightthickness=2)
        inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            inner, text=title,
            font=("Microsoft YaHei UI", 16, "bold"),
            fg="#e94560", bg="#16213e",
        ).pack(pady=(20, 2))

        self.status_label = tk.Label(
            inner, text=subtitle or "正在启动...",
            font=("Microsoft YaHei UI", 10),
            fg="#a0a0b0", bg="#16213e",
        )
        self.status_label.pack(pady=(0, 16))

        self.root.update()

    def set_status(self, text: str):
        try:
            self.status_label.config(text=text)
            self.root.update()
        except Exception:
            pass

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    import time
    s = SplashScreen("古诗词语音识别系统", "正在初始化 Python 环境...")
    for i in range(5, 0, -1):
        s.set_status(f"模拟加载中... {i} 秒后关闭")
        time.sleep(1)
    s.close()
    print("启动画面已关闭")
