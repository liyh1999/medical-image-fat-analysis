"""
GUI基础模块
包含GUI的基础框架和通用功能
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import logging
from utils import logger, natural_sort_key
from analyzer import FFImageAnalyzer, analyze_ff_image, batch_analyze_ff_images
from nifti_viewer import NIfTI3DViewer, load_nifti_image

class BaseGUI:
    """GUI基础类，包含通用功能"""
    
    def __init__(self, parent):
        self.parent = parent
        # 如果parent是Tkinter根窗口，直接使用
        if hasattr(parent, 'title'):
            self.root = parent
            self.root.title("界面化工具")
            self.root.geometry("1400x900")
        else:
            # 如果parent是Frame，需要找到根窗口
            self.root = parent.winfo_toplevel()
            if not hasattr(self.root, 'title'):
                self.root = parent
        
        # 模式变量
        
        # 数据变量
        self.ff_image = None
        self.roi_mask = None
        self.fat_fraction_values = None
        self.results = {}
        self.current_roi_type = 'rectangle'
        self.roi_type_var = tk.StringVar(value="rectangle")
        self.roi_list = []  # 当前图像的ROI列表
        self.image_roi_dict = {}  # 存储每张图像的ROI字典 {image_file: [roi_list]}
        self.debug_enabled = False  # 调试日志开关
        self.drawing = False
        
        # 图像显示相关
        self.canvas_width = 800
        self.canvas_height = 600
        self.scale_factor = 1.0
        self.image_tk = None
        self.image_offset_x = 0  # 图像在画布中的X偏移
        self.image_offset_y = 0  # 图像在画布中的Y偏移
        
        # 输出目录设置
        self.output_directory = None  # 保存目录，None表示未设置
        
        # 多边形ROI相关变量
        self.polygon_points = []  # 存储多边形顶点
        self.polygon_drawing = False  # 是否正在绘制多边形
        
        # 图像旋转控制
        self.rotation_angle = 0  # 当前旋转角度 (0, 90, 180, 270)
        
        # 鼠标绘制相关
        self.start_point = None
        self.end_point = None
        self.center_point = None
        self.radius = 0
        
        self.setup_base_ui()
        
    def setup_base_ui(self):
        """设置基础用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        # 创建工具栏
        self.create_toolbar(main_frame)
        
        # 创建主内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # 左侧图像显示区域
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 图像画布
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, 
                               bg='gray', cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # 绑定键盘事件
        self.root.bind("<Key>", self.on_key_press)
        self.root.focus_set()
        
        # 右侧控制面板
        self.create_control_panel(content_frame)
        
        
    def create_toolbar(self, parent):
        """创建工具栏"""
        self.toolbar = ttk.Frame(parent)
        self.toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # 子类应该重写此方法来创建特定的工具栏
        # 这里只创建基础的工具栏框架
        
    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 输出信息框
        output_frame = ttk.LabelFrame(control_frame, text="ROI计算结果")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.output_text = tk.Text(output_frame, height=20, width=30, wrap=tk.WORD, font=("Consolas", 9))
        output_scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # 快捷键帮助（缩小版）
        help_frame = ttk.LabelFrame(control_frame, text="快捷键")
        help_frame.pack(fill=tk.X, pady=(0, 0))
        self.shortcut_text = tk.Text(help_frame, height=8, width=30, wrap=tk.WORD, font=("Consolas", 8))
        shortcut_scrollbar = ttk.Scrollbar(help_frame, orient=tk.VERTICAL, command=self.shortcut_text.yview)
        self.shortcut_text.configure(yscrollcommand=shortcut_scrollbar.set)
        self.shortcut_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        shortcut_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # 初始化快捷键显示
        self.update_shortcut_display()
        # 初始化输出显示
        self.update_output_display()
    
    def update_output_display(self):
        """更新输出显示"""
        if hasattr(self, 'output_text') and self.output_text:
            self.output_text.delete(1.0, tk.END)
            
            if not self.roi_list:
                self.output_text.insert(tk.END, "暂无ROI数据\n\n请绘制ROI区域")
                return
            
            # 显示所有ROI的计算结果
            output_lines = []
            output_lines.append(f"ROI总数: {len(self.roi_list)}\n")
            output_lines.append("=" * 40 + "\n")
            
            for i, roi in enumerate(self.roi_list):
                output_lines.append(f"ROI #{i+1} ({roi['type']}):\n")
                
                if 'fat_fraction' in roi and roi['fat_fraction']:
                    ff = roi['fat_fraction']
                    
                    # 基本统计信息
                    output_lines.append("【基本统计】\n")
                    output_lines.append(f"  平均脂肪分数: {ff['mean_fat_fraction']:.3f}\n")
                    output_lines.append(f"  标准差: {ff['std_fat_fraction']:.3f}\n")
                    output_lines.append(f"  中位数: {ff['median_fat_fraction']:.3f}\n")
                    output_lines.append(f"  最小值: {ff['min_fat_fraction']:.3f}\n")
                    output_lines.append(f"  最大值: {ff['max_fat_fraction']:.3f}\n")
                    
                    # 分布信息
                    output_lines.append("\n【分布信息】\n")
                    range_val = ff['max_fat_fraction'] - ff['min_fat_fraction']
                    output_lines.append(f"  数值范围: {range_val:.3f}\n")
                    cv = (ff['std_fat_fraction'] / ff['mean_fat_fraction']) * 100 if ff['mean_fat_fraction'] > 0 else 0
                    output_lines.append(f"  变异系数: {cv:.2f}%\n")
                    
                    # 区域信息
                    output_lines.append("\n【区域信息】\n")
                    output_lines.append(f"  像素数量: {ff['pixel_count']:,}\n")
                    output_lines.append(f"  覆盖率: {ff['coverage_percentage']:.2f}%\n")
                    
                    # 质量评估
                    output_lines.append("\n【质量评估】\n")
                    if ff['std_fat_fraction'] < 0.05:
                        quality = "优秀"
                    elif ff['std_fat_fraction'] < 0.1:
                        quality = "良好"
                    elif ff['std_fat_fraction'] < 0.2:
                        quality = "一般"
                    else:
                        quality = "较差"
                    output_lines.append(f"  数据质量: {quality}\n")
                    
                    # 归一化信息
                    if ff.get('normalized', False):
                        output_lines.append(f"  归一化: 是\n")
                    else:
                        output_lines.append(f"  归一化: 否\n")
                else:
                    output_lines.append("  未计算脂肪分数\n")
                
                output_lines.append("=" * 40 + "\n")
            
            self.output_text.insert(tk.END, "".join(output_lines))
        
    
    def open_ff_image(self):
        """打开FF图像（基础版本）"""
        file_path = filedialog.askopenfilename(
            title="选择FF图像文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                from analyzer import FFImageAnalyzer
                analyzer = FFImageAnalyzer(file_path)
                self.ff_image = analyzer.ff_image
                self.display_image()
                if hasattr(self, 'status_var') and self.status_var:
                    self.status_var.set(f"已加载图像: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {str(e)}")
    
    def open_roi_mask(self):
        """打开ROI掩码（基础版本）"""
        file_path = filedialog.askopenfilename(
            title="选择ROI掩码文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                import cv2
                self.roi_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if hasattr(self, 'status_var') and self.status_var:
                    self.status_var.set(f"已加载ROI掩码: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"加载ROI掩码失败: {str(e)}")
    
    def display_image(self):
        """显示图像（基础版本）"""
        if self.ff_image is None:
            return
        
        # 应用旋转
        if self.rotation_angle != 0:
            rotated_image = self.apply_rotation(self.ff_image)
        else:
            rotated_image = self.ff_image.copy()
        
        # 转换为PIL图像
        if len(rotated_image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(rotated_image)
        
        # 计算缩放比例
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / rotated_image.shape[1]
            scale_y = canvas_height / rotated_image.shape[0]
            self.scale_factor = min(scale_x, scale_y, 1.0)
            
            # 计算图像在画布中的位置
            new_width = int(rotated_image.shape[1] * self.scale_factor)
            new_height = int(rotated_image.shape[0] * self.scale_factor)
            
            self.image_offset_x = (canvas_width - new_width) // 2
            self.image_offset_y = (canvas_height - new_height) // 2
            
            # 调整图像大小
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter图像
        self.image_tk = ImageTk.PhotoImage(pil_image)
        
        # 在画布上显示图像
        self.canvas.delete("all")
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, 
                               anchor=tk.NW, image=self.image_tk)
        
        # 绘制ROI
        self.draw_rois_on_canvas()
        
        # 更新输出显示
        self.update_output_display()
    
    def draw_rois_on_canvas(self):
        """在Tkinter画布上绘制ROI并显示脂肪分数"""
        logger.debug(f"绘制ROI，总数: {len(self.roi_list)}")
        for i, roi in enumerate(self.roi_list):
            logger.debug(f"ROI {i}: {roi['type']}, 脂肪分数数据: {'fat_fraction' in roi}")
            # 根据旋转角度调整坐标
            if self.rotation_angle != 0:
                if roi['type'] == 'rectangle':
                    start = self.rotate_roi_coordinates_for_display([roi['start']], self.rotation_angle)[0]
                    end = self.rotate_roi_coordinates_for_display([roi['end']], self.rotation_angle)[0]
                elif roi['type'] == 'circle':
                    center = self.rotate_roi_coordinates_for_display([roi['center']], self.rotation_angle)[0]
                    radius = roi['radius']
                elif roi['type'] == 'polygon':
                    points = self.rotate_roi_coordinates_for_display(roi['points'], self.rotation_angle)
            else:
                if roi['type'] == 'rectangle':
                    start = roi['start']
                    end = roi['end']
                elif roi['type'] == 'circle':
                    center = roi['center']
                    radius = roi['radius']
                elif roi['type'] == 'polygon':
                    points = roi['points']
            
            # 计算画布坐标
            if roi['type'] == 'rectangle':
                x1 = int(start[0] * self.scale_factor) + self.image_offset_x
                y1 = int(start[1] * self.scale_factor) + self.image_offset_y
                x2 = int(end[0] * self.scale_factor) + self.image_offset_x
                y2 = int(end[1] * self.scale_factor) + self.image_offset_y
                
                # 绘制矩形
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=2, tags=f"roi_{i}")
                
                # 计算中心点并显示脂肪分数
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
            elif roi['type'] == 'circle':
                cx = int(center[0] * self.scale_factor) + self.image_offset_x
                cy = int(center[1] * self.scale_factor) + self.image_offset_y
                r = int(radius * self.scale_factor)
                
                # 绘制圆形
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline='green', width=2, tags=f"roi_{i}")
                
                # 中心点就是圆心
                center_x = cx
                center_y = cy
                
            elif roi['type'] == 'polygon':
                # 转换多边形点到画布坐标
                canvas_points = []
                for point in points:
                    canvas_x = int(point[0] * self.scale_factor) + self.image_offset_x
                    canvas_y = int(point[1] * self.scale_factor) + self.image_offset_y
                    canvas_points.extend([canvas_x, canvas_y])
                
                # 绘制多边形
                self.canvas.create_polygon(canvas_points, outline='green', width=2, fill='', tags=f"roi_{i}")
                
                # 计算多边形中心点
                center_x = sum(canvas_points[::2]) // len(canvas_points[::2])
                center_y = sum(canvas_points[1::2]) // len(canvas_points[1::2])
            
            # 显示ROI标签和脂肪分数
            if 'fat_fraction' in roi and roi['fat_fraction']:
                ff_data = roi['fat_fraction']
                mean_ff = ff_data.get('mean_fat_fraction', 0)
                text = f"ROI{i+1}: {mean_ff:.3f}"
            else:
                text = f"ROI{i+1}"
            
            # 在ROI中心显示文本
            self.canvas.create_text(center_x, center_y, text=text, fill='yellow', 
                                  font=('Arial', 10, 'bold'), tags=f"roi_text_{i}")
        
    def clear_all_data(self):
        """清除所有数据"""
        self.ff_image = None
        self.roi_mask = None
        self.fat_fraction_values = None
        self.results = {}
        self.roi_list = []
        self.image_roi_dict = {}
        self.polygon_points = []
        self.polygon_drawing = False
        self.drawing = False
        self.rotation_angle = 0
        self.scale_factor = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        # 清空画布（如果存在）
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.delete("all")
        self.image_tk = None
        
        # 更新信息显示（如果存在）
        if hasattr(self, 'update_image_info'):
            self.update_image_info()
        if hasattr(self, 'update_roi_info'):
            self.update_roi_info()
        
    def update_shortcut_display(self):
        """更新快捷键显示"""
        shortcuts = [
            "通用快捷键:",
            "Ctrl+O: 打开图像",
            "Ctrl+S: 保存结果",
            "Ctrl+Q: 退出程序",
            "",
            "ROI绘制:",
            "R: 矩形ROI模式",
            "C: 圆形ROI模式", 
            "P: 多边形ROI模式",
            "S: 保存当前ROI",
            "Del: 删除最后ROI",
            "Ctrl+A: 清除所有ROI",
            "",
            "图像操作:",
            "Ctrl+R: 旋转90度",
            "Ctrl+Shift+R: 旋转270度",
            "Ctrl+0: 重置旋转",
            "",
            "计算:",
            "Enter: 计算脂肪分数",
            "Q: 实时计算",
            "Esc: 取消操作"
        ]
        
        if True:  # 默认使用交互式模式
            shortcuts.extend([
                "",
                "交互模式:",
                "←/A: 上一张图像",
                "→/D: 下一张图像",
                "Ctrl+Shift+S: 保存当前图像ROI"
            ])
        else:
            shortcuts.extend([
                "",
                "批量模式:",
                "←/A: 上一张图像",
                "→/D: 下一张图像",
                "Space: 开始/停止处理",
                "Ctrl+E: 导出结果"
            ])
        
        if hasattr(self, 'shortcut_text') and self.shortcut_text:
            self.shortcut_text.delete(1.0, tk.END)
            self.shortcut_text.insert(1.0, "\n".join(shortcuts))
        
    def on_key_press(self, event):
        """键盘事件处理"""
        if self.debug_enabled:
            logger.debug(f"按键事件: {event.keysym}")
        
        # 通用快捷键
        if event.state & 0x4:  # Ctrl键
            if event.keysym == "o":
                self.open_ff_image()
            elif event.keysym == "s":
                # 统一保存功能 - 根据模式调用不同的保存方法
                if hasattr(self, 'save_current_image_with_roi'):
                    self.save_current_image_with_roi()  # 交互式模式：保存图像+ROI文件
                elif hasattr(self, 'export_batch_results'):
                    self.export_batch_results()  # 批量模式：导出结果
            elif event.keysym == "q":
                self.on_closing()
            elif event.keysym == "r":
                self.rotate_image_90()
            elif event.keysym == "a":
                self.clear_all_roi()
            elif event.state & 0x1:  # Ctrl+Shift
                if event.keysym == "R":
                    self.rotate_image_270()
        elif event.keysym == "Escape":
            self.cancel_current_operation()
        elif event.keysym == "Return":
            self.final_calculation()
        elif event.keysym == "q":
            self.auto_calculate_and_display()
        elif event.keysym == "r":
            self.roi_type_var.set("rectangle")
            self.on_roi_type_change(None)
        elif event.keysym == "c":
            self.roi_type_var.set("circle")
            self.on_roi_type_change(None)
        elif event.keysym == "p":
            self.roi_type_var.set("polygon")
            self.on_roi_type_change(None)
        elif event.keysym == "s":
            self.save_current_roi()
        elif event.keysym == "Delete":
            self.delete_last_roi()
        elif event.keysym == "0":
            self.reset_rotation()
        else:
            # 子类可以重写此方法处理特定模式的快捷键
            self.handle_mode_specific_keys(event)
    
    def handle_mode_specific_keys(self, event):
        """处理特定模式的快捷键（子类重写）"""
        pass
        
    def cancel_current_operation(self):
        """取消当前操作"""
        if self.current_roi_type == 'polygon' and len(self.polygon_points) > 0:
            self.polygon_points = []
            self.polygon_drawing = False
            self.display_image()
            self.status_var.set("已取消多边形绘制")
        elif self.drawing:
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.center_point = None
            self.radius = 0
            self.display_image()
            self.status_var.set("已取消ROI绘制")
    
    def toggle_debug_logging(self):
        """切换调试日志"""
        self.debug_enabled = self.debug_var.get()
        if self.debug_enabled:
            logger.setLevel(logging.DEBUG)
            logger.info("调试日志已启用")
        else:
            logger.setLevel(logging.INFO)
            logger.info("调试日志已禁用")
    
    def update_image_info(self):
        """更新图像信息显示"""
        if hasattr(self, 'image_info_var') and self.image_info_var:
            if self.ff_image is not None:
                height, width = self.ff_image.shape[:2]
                info_text = f"尺寸: {width}x{height}\n类型: {self.ff_image.dtype}\n旋转: {self.rotation_angle}°"
                self.image_info_var.set(info_text)
            else:
                self.image_info_var.set("未加载图像")
    
    def update_roi_info(self):
        """更新ROI信息显示"""
        if hasattr(self, 'roi_info_var') and self.roi_info_var:
            if self.roi_list:
                total_area = 0
                for roi in self.roi_list:
                    if roi['type'] == 'rectangle':
                        x1, y1 = roi['start']
                        x2, y2 = roi['end']
                        area = abs((x2-x1) * (y2-y1))
                    elif roi['type'] == 'circle':
                        area = 3.14159 * roi['radius'] * roi['radius']
                    elif roi['type'] == 'polygon':
                        if len(roi['points']) >= 3:
                            points = np.array(roi['points'], dtype=np.float32)
                            area = cv2.contourArea(points)
                        else:
                            area = 0
                    total_area += area
                
                info_text = f"ROI数量: {len(self.roi_list)}\n总面积: {total_area:.0f} 像素"
                self.roi_info_var.set(info_text)
            else:
                self.roi_info_var.set("无ROI")
    
    def on_mouse_click(self, event):
        """鼠标点击事件"""
        if self.ff_image is None:
            return
            
        if self.current_roi_type == 'polygon':
            # 多边形模式：添加顶点
            # 考虑图像偏移量
            adjusted_x = event.x - self.image_offset_x
            adjusted_y = event.y - self.image_offset_y
            point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
            self.polygon_points.append(point)
            
            # 检查是否应该闭合多边形（至少3个点且点击位置接近第一个点）
            if len(self.polygon_points) >= 3:
                first_point = self.polygon_points[0]
                distance = np.sqrt((point[0] - first_point[0])**2 + (point[1] - first_point[1])**2)
                if distance < 20:  # 20像素范围内认为点击了第一个点
                    # 闭合多边形
                    self.polygon_drawing = True
                    self.save_polygon_roi()
                    return
            
            # 更新显示
            self.display_image()
            self.status_var.set(f"多边形模式：已添加{len(self.polygon_points)}个顶点，点击首尾两点闭合")
        else:
            # 矩形和圆形模式
            self.drawing = True
            # 考虑图像偏移量
            adjusted_x = event.x - self.image_offset_x
            adjusted_y = event.y - self.image_offset_y
            self.start_point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
            self.end_point = self.start_point
            self.center_point = self.start_point
            self.radius = 0
    
    def on_mouse_drag(self, event):
        """鼠标拖拽事件"""
        if not self.drawing or self.ff_image is None:
            return
            
        # 考虑图像偏移量
        adjusted_x = event.x - self.image_offset_x
        adjusted_y = event.y - self.image_offset_y
        current_point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
        
        if self.current_roi_type == 'rectangle':
            self.end_point = current_point
        elif self.current_roi_type == 'circle':
            self.center_point = current_point
            self.radius = int(np.sqrt((current_point[0] - self.start_point[0])**2 + 
                                    (current_point[1] - self.start_point[1])**2))
        
        # 绘制当前正在绘制的ROI
        self.draw_current_roi()
    
    def on_mouse_release(self, event):
        """鼠标释放事件"""
        if not self.drawing or self.ff_image is None:
            return
            
        self.drawing = False
        # 考虑图像偏移量
        adjusted_x = event.x - self.image_offset_x
        adjusted_y = event.y - self.image_offset_y
        current_point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
        
        if self.current_roi_type == 'rectangle':
            self.end_point = current_point
        elif self.current_roi_type == 'circle':
            self.center_point = current_point
            self.radius = int(np.sqrt((current_point[0] - self.start_point[0])**2 + 
                                    (current_point[1] - self.start_point[1])**2))
        
        # 绘制当前正在绘制的ROI（不自动保存，需要按S键保存）
        self.draw_current_roi()
    
    def auto_save_and_calculate_roi(self):
        """自动保存并计算ROI"""
        if self.current_roi_type == 'rectangle' and self.start_point and self.end_point:
            # 保存矩形ROI
            roi = {
                'type': 'rectangle',
                'start': self.start_point,
                'end': self.end_point
            }
            self.roi_list.append(roi)
            
            # 计算脂肪分数
            self.calculate_roi_fat_fraction(roi)
            
            # 清空当前绘制状态
            self.start_point = None
            self.end_point = None
            
        elif self.current_roi_type == 'circle' and self.center_point and self.radius > 0:
            # 保存圆形ROI
            roi = {
                'type': 'circle',
                'center': self.center_point,
                'radius': self.radius
            }
            self.roi_list.append(roi)
            
            # 计算脂肪分数
            self.calculate_roi_fat_fraction(roi)
            
            # 清空当前绘制状态
            self.center_point = None
            self.radius = 0
        
        # 更新输出显示
        self.update_output_display()
    
    def calculate_roi_fat_fraction(self, roi):
        """计算单个ROI的脂肪分数"""
        try:
            # 创建ROI掩码
            roi_mask = self.create_roi_mask(roi)
            if roi_mask is None:
                return
            
            # 计算脂肪分数
            from analyzer import FFImageAnalyzer
            analyzer = FFImageAnalyzer("dummy_path")
            analyzer.ff_image = self.ff_image
            analyzer.roi_mask = roi_mask
            
            result = analyzer.calculate_fat_fraction()
            
            # 将结果保存到ROI中，确保包含所有必要字段
            roi['fat_fraction'] = {
                'mean_fat_fraction': result.get('mean_fat_fraction', 0),
                'std_fat_fraction': result.get('std_fat_fraction', 0),
                'median_fat_fraction': result.get('median_fat_fraction', 0),
                'min_fat_fraction': result.get('min_fat_fraction', 0),
                'max_fat_fraction': result.get('max_fat_fraction', 0),
                'pixel_count': result.get('pixel_count', 0),
                'coverage_percentage': result.get('coverage_percentage', 0),
                'normalized': result.get('normalized', False)
            }
            
            logger.info(f"ROI {roi['type']} 脂肪分数计算完成: {result['mean_fat_fraction']:.3f}")
            
        except Exception as e:
            logger.error(f"计算ROI脂肪分数失败: {str(e)}")
    
    def create_roi_mask(self, roi):
        """根据ROI信息创建掩码"""
        if self.ff_image is None:
            return None
        
        height, width = self.ff_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        try:
            if roi['type'] == 'rectangle':
                x1, y1 = roi['start']
                x2, y2 = roi['end']
                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 边界检查
                x1 = max(0, min(x1, width-1))
                x2 = max(0, min(x2, width-1))
                y1 = max(0, min(y1, height-1))
                y2 = max(0, min(y2, height-1))
                
                # 确保有有效的区域
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
                else:
                    logger.warning("矩形ROI区域无效")
                    return None
                    
            elif roi['type'] == 'circle':
                cx, cy = roi['center']
                radius = roi['radius']
                
                # 边界检查
                cx = max(0, min(cx, width-1))
                cy = max(0, min(cy, height-1))
                radius = max(1, min(radius, min(width, height)//2))
                
                cv2.circle(mask, (cx, cy), radius, 255, -1)
                
            elif roi['type'] == 'polygon':
                points = np.array(roi['points'], dtype=np.int32)
                
                # 边界检查
                points[:, 0] = np.clip(points[:, 0], 0, width-1)
                points[:, 1] = np.clip(points[:, 1], 0, height-1)
                
                cv2.fillPoly(mask, [points], 255)
            
            # 检查掩码是否有效
            if np.sum(mask) == 0:
                logger.warning("ROI区域内没有有效的像素值")
                return None
                
            return mask
            
        except Exception as e:
            logger.error(f"创建ROI掩码失败: {str(e)}")
            return None
    
    def draw_rois_on_image(self, image):
        """在图像上绘制ROI"""
        for i, roi in enumerate(self.roi_list):
            # 根据旋转角度调整坐标
            if self.rotation_angle != 0:
                if roi['type'] == 'rectangle':
                    start = self.rotate_roi_coordinates_for_display([roi['start']], self.rotation_angle)[0]
                    end = self.rotate_roi_coordinates_for_display([roi['end']], self.rotation_angle)[0]
                elif roi['type'] == 'circle':
                    center = self.rotate_roi_coordinates_for_display([roi['center']], self.rotation_angle)[0]
                    radius = roi['radius']
                elif roi['type'] == 'polygon':
                    points = self.rotate_roi_coordinates_for_display(roi['points'], self.rotation_angle)
            else:
                if roi['type'] == 'rectangle':
                    start = roi['start']
                    end = roi['end']
                elif roi['type'] == 'circle':
                    center = roi['center']
                    radius = roi['radius']
                elif roi['type'] == 'polygon':
                    points = roi['points']
            
            # 绘制ROI
            color = (0, 255, 0)  # 绿色
            thickness = 2
            
            if roi['type'] == 'rectangle':
                x1, y1 = int(start[0]), int(start[1])
                x2, y2 = int(end[0]), int(end[1])
                
                # 确保坐标在图像范围内
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # 添加标签
                text_x = min(x1, x2)
                text_y = min(y1, y2) - 5
                label_text = f'ROI{i+1}'
                if roi.get('fat_fraction') is not None:
                    label_text += f' FF:{roi["fat_fraction"]:.3f}'
                cv2.putText(image, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            elif roi['type'] == 'circle':
                cx = int(center[0])
                cy = int(center[1])
                r = int(radius)
                
                # 确保坐标在图像范围内
                h, w = image.shape[:2]
                cx = max(0, min(cx, w-1))
                cy = max(0, min(cy, h-1))
                r = max(1, min(r, min(w, h)//2))
                
                cv2.circle(image, (cx, cy), r, color, thickness)
                
                # 添加标签
                text_x = cx - 20
                text_y = cy - r - 5
                label_text = f'ROI{i+1}'
                if roi.get('fat_fraction') is not None:
                    label_text += f' FF:{roi["fat_fraction"]:.3f}'
                cv2.putText(image, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            elif roi['type'] == 'polygon':
                # 绘制多边形
                h, w = image.shape[:2]
                # 确保所有点都在图像范围内
                valid_points = []
                for p in points:
                    x, y = int(p[0]), int(p[1])
                    x = max(0, min(x, w-1))
                    y = max(0, min(y, h-1))
                    valid_points.append((x, y))
                
                points = np.array(valid_points, dtype=np.int32)
                cv2.polylines(image, [points], True, color, thickness)
                
                # 计算文本位置（使用多边形的中心点）
                center_x = int(np.mean([p[0] for p in points]))
                center_y = int(np.mean([p[1] for p in points]))
                text_x = max(5, center_x - 30)
                text_y = max(25, center_y - 5)
                label_text = f'ROI{i+1}'
                if roi.get('fat_fraction') is not None:
                    label_text += f' FF:{roi["fat_fraction"]:.3f}'
                cv2.putText(image, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    def draw_temp_polygon(self, image):
        """绘制临时多边形（正在绘制的多边形）"""
        if len(self.polygon_points) < 2:
            return
        
        # 根据旋转角度调整坐标
        if self.rotation_angle != 0:
            temp_points = self.rotate_roi_coordinates_for_display(self.polygon_points, self.rotation_angle)
        else:
            temp_points = self.polygon_points
        
        # 绘制已添加的点
        for i, point in enumerate(temp_points):
            x = int(point[0])
            y = int(point[1])
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)  # 蓝色圆点
            cv2.putText(image, str(i+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制连接线
        points = np.array([(int(p[0]), int(p[1])) for p in temp_points], dtype=np.int32)
        cv2.polylines(image, [points], False, (255, 0, 0), 2)  # 蓝色线条
        
        # 如果至少3个点，绘制闭合提示线
        if len(temp_points) >= 3:
            first_point = (int(temp_points[0][0]), int(temp_points[0][1]))
            last_point = (int(temp_points[-1][0]), int(temp_points[-1][1]))
            cv2.line(image, first_point, last_point, (0, 255, 255), 1)  # 黄色虚线提示闭合
    
    def draw_current_roi(self):
        """绘制当前正在绘制的ROI"""
        if not self.drawing or self.ff_image is None:
            return
            
        # 应用旋转
        if self.rotation_angle != 0:
            rotated_image = self.apply_rotation(self.ff_image)
        else:
            rotated_image = self.ff_image.copy()
        
        # 创建临时图像用于绘制
        temp_image = rotated_image.copy()
        if len(temp_image.shape) == 2:
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
        
        # 绘制当前ROI（需要根据旋转调整坐标）
        if self.current_roi_type == 'rectangle' and self.start_point and self.end_point:
            # 根据旋转角度调整坐标
            if self.rotation_angle != 0:
                start = self.rotate_roi_coordinates_for_display([self.start_point], self.rotation_angle)[0]
                end = self.rotate_roi_coordinates_for_display([self.end_point], self.rotation_angle)[0]
            else:
                start = self.start_point
                end = self.end_point
            
            x1 = int(start[0])
            y1 = int(start[1])
            x2 = int(end[0])
            y2 = int(end[1])
            cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色
        elif self.current_roi_type == 'circle' and self.center_point and self.radius > 0:
            # 根据旋转角度调整坐标
            if self.rotation_angle != 0:
                center = self.rotate_roi_coordinates_for_display([self.center_point], self.rotation_angle)[0]
            else:
                center = self.center_point
            
            cx = int(center[0])
            cy = int(center[1])
            r = int(self.radius)
            cv2.circle(temp_image, (cx, cy), r, (0, 0, 255), 2)  # 红色
        
        # 绘制已保存的ROI
        self.draw_rois_on_image(temp_image)
        
        # 绘制临时多边形
        if self.current_roi_type == 'polygon' and len(self.polygon_points) > 0:
            self.draw_temp_polygon(temp_image)
        
        # 转换为PIL图像
        if len(temp_image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(temp_image)
        
        # 计算缩放比例
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / temp_image.shape[1]
            scale_y = canvas_height / temp_image.shape[0]
            self.scale_factor = min(scale_x, scale_y, 1.0)
            
            # 计算图像在画布中的位置
            new_width = int(temp_image.shape[1] * self.scale_factor)
            new_height = int(temp_image.shape[0] * self.scale_factor)
            
            self.image_offset_x = (canvas_width - new_width) // 2
            self.image_offset_y = (canvas_height - new_height) // 2
            
            # 调整图像大小
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter图像
        self.image_tk = ImageTk.PhotoImage(pil_image)
        
        # 在画布上显示图像
        self.canvas.delete("all")
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, 
                               anchor=tk.NW, image=self.image_tk)
    
    def rotate_roi_coordinates_for_display(self, points, rotation_angle):
        """为显示旋转ROI坐标"""
        if rotation_angle == 0:
            return points
        
        # 获取原始图像尺寸
        if self.ff_image is not None:
            h, w = self.ff_image.shape[:2]
        else:
            return points
        
        rotated_points = []
        for point in points:
            x, y = point
            
            # 根据旋转角度调整坐标
            if rotation_angle == 90:
                # 顺时针90度：(x,y) -> (h-1-y, x)
                # 旋转后图像尺寸变为 (w, h)
                new_x = h - 1 - y
                new_y = x
            elif rotation_angle == 180:
                # 180度：(x,y) -> (w-1-x, h-1-y)
                new_x = w - 1 - x
                new_y = h - 1 - y
            elif rotation_angle == 270:
                # 顺时针270度：(x,y) -> (y, w-1-x)
                # 旋转后图像尺寸变为 (w, h)
                new_x = y
                new_y = w - 1 - x
            else:
                new_x, new_y = x, y
            
            rotated_points.append((new_x, new_y))
        
        return rotated_points
    
    def apply_rotation(self, image, angle):
        """应用图像旋转"""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    def rotate_image_90(self):
        """旋转图像90度"""
        if self.ff_image is not None:
            self.rotation_angle = (self.rotation_angle + 90) % 360
            self.display_image()
            self.status_var.set(f"图像已旋转 {self.rotation_angle}°")
    
    def rotate_image_270(self):
        """旋转图像270度"""
        if self.ff_image is not None:
            self.rotation_angle = (self.rotation_angle + 270) % 360
            self.display_image()
            self.status_var.set(f"图像已旋转 {self.rotation_angle}°")
    
    def reset_rotation(self):
        """重置旋转"""
        if self.ff_image is not None:
            self.rotation_angle = 0
            self.display_image()
            self.status_var.set("图像旋转已重置")
    
    def on_roi_type_change(self, event):
        """ROI类型改变事件"""
        self.current_roi_type = self.roi_type_var.get()
        
        # 如果切换到多边形模式，清空之前的绘制状态
        if self.current_roi_type == 'polygon':
            self.polygon_points = []
            self.polygon_drawing = False
            self.drawing = False
            self.status_var.set("多边形模式：点击绘制顶点，首尾两点自动闭合")
        else:
            self.polygon_points = []
            self.polygon_drawing = False
            self.status_var.set(f"已切换到{self.current_roi_type}模式")
    
    def save_polygon_roi(self):
        """保存多边形ROI"""
        if len(self.polygon_points) < 3:
            messagebox.showwarning("警告", "多边形至少需要3个顶点")
            return
        
        # 创建多边形ROI
        roi = {
            'type': 'polygon',
            'points': self.polygon_points.copy()
        }
        
        # 计算脂肪分数
        try:
            fat_fraction = self.calculate_single_roi_fat_fraction(roi)
            roi['fat_fraction'] = fat_fraction
        except Exception as e:
            logger.warning(f"计算多边形ROI脂肪分数失败: {str(e)}")
            roi['fat_fraction'] = None
        
        # 添加到ROI列表
        self.roi_list.append(roi)
        
        # 清空多边形绘制状态
        self.polygon_points = []
        self.polygon_drawing = False
        
        # 更新显示
        self.display_image()
        self.update_roi_info()
        
        # 自动计算并显示结果
        self.auto_calculate_and_display()
        
        self.status_var.set(f"已保存多边形ROI，包含{len(roi['points'])}个顶点")
    
    def save_current_roi(self):
        """保存当前ROI"""
        if self.current_roi_type == 'polygon':
            # 多边形ROI通过save_polygon_roi处理
            if len(self.polygon_points) >= 3:
                self.save_polygon_roi()
            else:
                messagebox.showwarning("警告", "多边形至少需要3个顶点")
            return
        
        # 矩形和圆形ROI
        if self.current_roi_type == 'rectangle' and self.start_point and self.end_point:
            roi = {
                'type': 'rectangle',
                'start': self.start_point,
                'end': self.end_point
            }
        elif self.current_roi_type == 'circle' and self.center_point and self.radius > 0:
            roi = {
                'type': 'circle',
                'center': self.center_point,
                'radius': self.radius
            }
        else:
            messagebox.showwarning("警告", "请先绘制一个ROI")
            return
        
        # 计算脂肪分数
        try:
            fat_fraction_data = self.calculate_single_roi_fat_fraction(roi)
            roi['fat_fraction'] = fat_fraction_data
            logger.info(f"ROI {roi['type']} 脂肪分数计算完成: {fat_fraction_data['mean_fat_fraction']:.3f}")
        except Exception as e:
            logger.warning(f"计算ROI脂肪分数失败: {str(e)}")
            roi['fat_fraction'] = None
        
        # 添加到ROI列表
        self.roi_list.append(roi)
        
        # 清空绘制状态
        self.start_point = None
        self.end_point = None
        self.center_point = None
        self.radius = 0
        self.drawing = False
        
        # 更新显示
        self.display_image()
        self.update_output_display()
        
        if hasattr(self, 'status_var') and self.status_var:
            self.status_var.set(f"已保存{self.current_roi_type}ROI")
    
    def calculate_single_roi_fat_fraction(self, roi):
        """计算单个ROI的脂肪分数"""
        if self.ff_image is None:
            raise ValueError("未加载图像")
        
        # 创建临时ROI掩码
        temp_mask = np.zeros(self.ff_image.shape, dtype=np.uint8)
        
        if roi['type'] == 'rectangle':
            x1, y1 = roi['start']
            x2, y2 = roi['end']
            temp_mask[y1:y2, x1:x2] = 255
        elif roi['type'] == 'circle':
            cv2.circle(temp_mask, roi['center'], roi['radius'], 255, -1)
        elif roi['type'] == 'polygon':
            points = np.array(roi['points'], dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 255)
        
        # 计算脂肪分数
        return self.calculate_fat_fraction_for_mask(temp_mask)
    
    def calculate_fat_fraction_for_mask(self, roi_mask):
        """根据掩码计算脂肪分数"""
        if self.ff_image is None:
            raise ValueError("未加载图像")
        
        # 检测图像像素值范围并决定是否需要归一化
        min_pixel = np.min(self.ff_image)
        max_pixel = np.max(self.ff_image)
        
        # 判断是否需要归一化
        if max_pixel <= 100 and min_pixel >= 0:
            normalized_ff_image = self.ff_image.astype(np.float32)
        else:
            normalized_ff_image = self.ff_image.astype(np.float32) / max_pixel
        
        # 确保ROI掩码是二值的
        roi_binary = (roi_mask > 0).astype(np.uint8)
        
        # 提取ROI区域内的FF值
        roi_ff_values = normalized_ff_image[roi_binary == 1]
        
        if len(roi_ff_values) == 0:
            raise ValueError("ROI区域内没有有效的像素值")
        
        # 计算统计信息
        mean_ff = np.mean(roi_ff_values)
        std_ff = np.std(roi_ff_values)
        median_ff = np.median(roi_ff_values)
        min_ff = np.min(roi_ff_values)
        max_ff = np.max(roi_ff_values)
        pixel_count = len(roi_ff_values)
        
        # 计算覆盖率（ROI面积占图像总面积的比例）
        total_pixels = self.ff_image.shape[0] * self.ff_image.shape[1]
        coverage_percentage = (pixel_count / total_pixels) * 100
        
        # 判断是否归一化
        normalized = not (max_pixel <= 100 and min_pixel >= 0)
        
        return {
            'mean_fat_fraction': float(mean_ff),
            'std_fat_fraction': float(std_ff),
            'median_fat_fraction': float(median_ff),
            'min_fat_fraction': float(min_ff),
            'max_fat_fraction': float(max_ff),
            'pixel_count': int(pixel_count),
            'coverage_percentage': float(coverage_percentage),
            'normalized': bool(normalized)
        }
    
    def auto_calculate_and_display(self):
        """自动计算并显示结果"""
        if not self.roi_list:
            return
        
        try:
            # 合并所有ROI
            combined_mask = np.zeros(self.ff_image.shape, dtype=np.uint8)
            for roi in self.roi_list:
                if roi['type'] == 'rectangle':
                    x1, y1 = roi['start']
                    x2, y2 = roi['end']
                    combined_mask[y1:y2, x1:x2] = 255
                elif roi['type'] == 'circle':
                    cv2.circle(combined_mask, roi['center'], roi['radius'], 255, -1)
                elif roi['type'] == 'polygon':
                    points = np.array(roi['points'], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [points], 255)
            
            # 计算脂肪分数
            result = self.calculate_fat_fraction_for_mask(combined_mask)
            
            # 显示结果
            self.display_results({'mean_fat_fraction': result})
            
        except Exception as e:
            logger.error(f"自动计算失败: {str(e)}")
            self.status_var.set(f"计算失败: {str(e)}")
    
    def delete_last_roi(self):
        """删除最后一个ROI"""
        if self.roi_list:
            self.roi_list.pop()
            self.display_image()
            self.update_roi_info()
            self.auto_calculate_and_display()
            self.status_var.set("已删除最后一个ROI")
        else:
            self.status_var.set("没有ROI可删除")
    
    def clear_all_roi(self):
        """清除所有ROI"""
        if self.roi_list:
            self.roi_list.clear()
            self.display_image()
            self.update_roi_info()
            self.status_var.set("已清除所有ROI")
        else:
            self.status_var.set("没有ROI可清除")
    
    def final_calculation(self):
        """最终计算"""
        if not self.roi_list:
            messagebox.showwarning("警告", "请先绘制ROI")
            return
        
        try:
            # 合并所有ROI
            combined_mask = np.zeros(self.ff_image.shape, dtype=np.uint8)
            for roi in self.roi_list:
                if roi['type'] == 'rectangle':
                    x1, y1 = roi['start']
                    x2, y2 = roi['end']
                    combined_mask[y1:y2, x1:x2] = 255
                elif roi['type'] == 'circle':
                    cv2.circle(combined_mask, roi['center'], roi['radius'], 255, -1)
                elif roi['type'] == 'polygon':
                    points = np.array(roi['points'], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [points], 255)
            
            # 计算脂肪分数
            result = self.calculate_fat_fraction_for_mask(combined_mask)
            
            # 显示结果
            self.display_results({'mean_fat_fraction': result})
            
        except Exception as e:
            messagebox.showerror("错误", f"计算失败: {str(e)}")
            logger.error(f"最终计算失败: {str(e)}")
    
    def display_results(self, result):
        """显示计算结果"""
        if 'mean_fat_fraction' in result:
            self.status_var.set(f"平均脂肪分数: {result['mean_fat_fraction']:.3f}")
        else:
            self.status_var.set("计算完成")
    
    def on_closing(self):
        """程序关闭时的清理工作"""
        logger.info("程序正在关闭，清理ROI数据...")
        # 清除所有ROI数据
        self.roi_list.clear()
        self.results = {}
        logger.info("ROI数据已清理完成")
        # 关闭窗口
        self.root.destroy()
    
    def set_output_directory(self):
        """设置输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_directory = directory
            self.status_var.set(f"输出目录已设置为: {directory}")
    
    def convert_mask_to_roi_list(self, roi_mask):
        """将掩码转换为ROI列表"""
        # 查找轮廓
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_list = []
        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < 100:  # 过滤面积小于100像素的区域
                continue
            
            # 创建多边形ROI
            roi = {
                'type': 'polygon',
                'points': contour.reshape(-1, 2).tolist(),
                'fat_fraction': None
            }
            roi_list.append(roi)
        
        logger.info(f"从掩码转换得到 {len(roi_list)} 个ROI")
        return roi_list
    
    def find_corresponding_label(self, image_file):
        """查找对应的标签文件"""
        if not hasattr(self, 'batch_labels_dir') or not self.batch_labels_dir:
            return None
        
        # 根据文件类型选择扩展名
        if image_file.lower().endswith(('.nii', '.nii.gz')):
            label_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.nii', '.nii.gz']
        else:
            label_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        
        # 1. 尝试完全匹配
        base_name = os.path.splitext(image_file)[0]
        logger.debug(f"开始匹配标签文件，图像文件: {image_file}, 基础名称: {base_name}")
        for ext in label_extensions:
            label_file = f"{base_name}_roi{ext}"
            label_path = os.path.join(self.batch_labels_dir, label_file)
            logger.debug(f"尝试完全匹配: {label_file} -> {label_path} (存在: {os.path.exists(label_path)})")
            if os.path.exists(label_path):
                logger.debug(f"完全匹配成功: {label_path}")
                return label_path
        
        # 2. 尝试去掉数字后缀匹配（如 _0000, _0001 等）
        import re
        # 匹配文件名末尾的数字后缀
        pattern = r'(_\d+)$'
        match = re.search(pattern, base_name)
        if match:
            # 去掉数字后缀
            clean_base_name = base_name[:match.start()]
            logger.debug(f"检测到数字后缀，清理后的名称: {clean_base_name}")
            
            # 2a. 尝试带_roi后缀的匹配
            for ext in label_extensions:
                label_file = f"{clean_base_name}_roi{ext}"
                label_path = os.path.join(self.batch_labels_dir, label_file)
                logger.debug(f"尝试带_roi后缀匹配: {label_file} -> {label_path} (存在: {os.path.exists(label_path)})")
                if os.path.exists(label_path):
                    logger.debug(f"带_roi后缀匹配成功: {label_path}")
                    return label_path
            
            # 2b. 尝试直接匹配（不带_roi后缀）
            for ext in label_extensions:
                label_file = f"{clean_base_name}{ext}"
                label_path = os.path.join(self.batch_labels_dir, label_file)
                logger.debug(f"尝试直接匹配: {label_file} -> {label_path} (存在: {os.path.exists(label_path)})")
                if os.path.exists(label_path):
                    logger.debug(f"直接匹配成功: {label_path}")
                    return label_path
        else:
            logger.debug(f"未检测到数字后缀，跳过第二级匹配")
        
        # 3. 尝试模糊匹配
        return self.fuzzy_match_label(image_file, label_extensions)
    
    def fuzzy_match_label(self, image_file, label_extensions):
        """模糊匹配标签文件"""
        if not hasattr(self, 'batch_labels_dir') or not self.batch_labels_dir:
            return None
        
        base_name = os.path.splitext(image_file)[0]
        
        # 获取所有标签文件
        label_files = []
        for file in os.listdir(self.batch_labels_dir):
            if any(file.lower().endswith(ext) for ext in label_extensions):
                label_files.append(file)
        
        # 尝试模糊匹配
        for label_file in label_files:
            label_base = os.path.splitext(label_file)[0]
            
            # 检查是否包含图像文件名
            if base_name in label_base or label_base in base_name:
                return os.path.join(self.batch_labels_dir, label_file)
        
        return None
    
    def update_image_index(self):
        """更新图像索引显示"""
        if hasattr(self, 'interactive_image_files') and self.interactive_image_files:
            self.image_index_var.set(f"{self.interactive_current_index + 1}/{len(self.interactive_image_files)}")
        else:
            self.image_index_var.set("0/0")
    
    def run(self):
        """运行应用程序"""
        # 设置窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
