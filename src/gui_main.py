"""
主GUI模块
整合所有GUI子模块，提供统一的界面入口
"""

import tkinter as tk
from tkinter import ttk, messagebox
from utils import logger
from gui_base import BaseGUI
from gui_interactive_2d import Interactive2DGUI
from gui_interactive_3d import Interactive3DGUI
from gui_batch_2d import Batch2DGUI
from gui_batch_3d import Batch3DGUI

class MainGUI:
    """主GUI类，管理不同的GUI模式"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.current_gui = None
        self.setup_main_ui()
        
    def setup_main_ui(self):
        """设置主界面"""
        self.root.title("勾画计算界面化软件 v2.1")
        self.root.geometry("1400x900")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建模式选择区域
        self.create_mode_selector(main_frame)
        
        # 创建内容区域
        self.content_frame = ttk.Frame(main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 默认显示交互式2D模式
        self.switch_to_interactive_2d()
        
    def create_mode_selector(self, parent):
        """创建模式选择器"""
        mode_frame = ttk.LabelFrame(parent, text="选择工作模式")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建模式选择按钮
        button_frame = ttk.Frame(mode_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="交互式2D模式", 
                  command=self.switch_to_interactive_2d).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="交互式3D模式", 
                  command=self.switch_to_interactive_3d).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="批量处理2D模式", 
                  command=self.switch_to_batch_2d).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="批量处理3D模式", 
                  command=self.switch_to_batch_3d).pack(side=tk.LEFT, padx=(0, 10))
        
    
    def clear_content(self):
        """清空内容区域"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def switch_to_interactive_2d(self):
        """切换到交互式2D模式"""
        self.clear_content()
        self.current_gui = Interactive2DGUI(self.content_frame)
        logger.info("切换到交互式2D模式")
    
    def switch_to_interactive_3d(self):
        """切换到交互式3D模式"""
        self.clear_content()
        self.current_gui = Interactive3DGUI(self.content_frame)
        logger.info("切换到交互式3D模式")
    
    def switch_to_batch_2d(self):
        """切换到批量处理2D模式"""
        self.clear_content()
        self.current_gui = Batch2DGUI(self.content_frame)
        logger.info("切换到批量处理2D模式")
    
    def switch_to_batch_3d(self):
        """切换到批量处理3D模式"""
        self.clear_content()
        self.current_gui = Batch3DGUI(self.content_frame)
        logger.info("切换到批量处理3D模式")
    
    def run(self):
        """运行主程序"""
        # 设置窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """程序关闭时的清理工作"""
        logger.info("程序正在关闭...")
        
        # 清理当前GUI
        if self.current_gui and hasattr(self.current_gui, 'on_closing'):
            self.current_gui.on_closing()
        
        # 关闭主窗口
        try:
            self.root.destroy()
        except Exception as e:
            logger.debug(f"关闭主窗口时出现异常: {str(e)}")

def main():
    """主函数"""
    print("=== 勾画计算界面化软件 v2.1 ===")
    print("启动图形界面...")
    
    # 创建并运行主GUI
    app = MainGUI()
    app.run()

if __name__ == "__main__":
    main()
