"""
批量处理3D模式GUI模块
处理批量3D NIfTI文件处理和分析
"""

import os
import cv2
import numpy as np
import tkinter as tk
import logging
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from utils import logger, natural_sort_key, NIBABEL_AVAILABLE
from gui_base import BaseGUI
from nifti_viewer import NIfTI3DViewer, load_nifti_image

class Batch3DGUI(BaseGUI):
    """批量处理3D模式GUI"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # 批量处理3D控制变量
        self.batch_3d_viewer = None  # 批量处理3D查看器
        self.batch_current_view = 'axial'  # 批量处理当前视图
        self.batch_current_slice = 0  # 批量处理当前切片
        
        # 状态变量
        self.status_var = tk.StringVar(value="批量处理3D模式：请选择图像文件夹")
    
    def display_image(self):
        """显示批量处理3D图像切片"""
        if self.batch_3d_viewer is None:
            return
        
        try:
            # 获取当前切片
            slice_data = self.batch_3d_viewer.get_slice(self.batch_current_view, self.batch_current_slice)
            if slice_data is None:
                return
            
            # 转换为PIL图像
            if len(slice_data.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(slice_data, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(slice_data)
            
            # 计算缩放比例
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                scale_x = canvas_width / slice_data.shape[1]
                scale_y = canvas_height / slice_data.shape[0]
                self.scale_factor = min(scale_x, scale_y, 1.0)
                
                # 计算图像在画布中的位置
                new_width = int(slice_data.shape[1] * self.scale_factor)
                new_height = int(slice_data.shape[0] * self.scale_factor)
                
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
            self.draw_rois_on_image()
            
        except Exception as e:
            logger.error(f"显示批量处理3D切片失败: {str(e)}")
            if hasattr(self, 'status_var') and self.status_var:
                self.status_var.set(f"显示切片失败: {str(e)}")
        self.batch_current_image_index = 0  # 当前显示的图像索引
        
        # 批量处理相关
        self.batch_images_dir = None
        self.batch_labels_dir = None
        self.batch_image_files = []
        self.batch_results = []
        self.batch_processing = False
        
    def create_toolbar(self, parent):
        """创建批量处理3D模式工具栏"""
        # 调用父类方法创建工具栏框架
        super().create_toolbar(parent)
        
        # 3D视图控制
        ttk.Label(self.toolbar, text="视图:").pack(side=tk.LEFT, padx=(0, 5))
        self.batch_view_var = tk.StringVar(value="axial")
        view_combo = ttk.Combobox(self.toolbar, textvariable=self.batch_view_var,
                                 values=["axial", "sagittal", "coronal"], state="readonly", width=10)
        view_combo.pack(side=tk.LEFT, padx=(0, 5))
        view_combo.bind("<<ComboboxSelected>>", self.on_batch_view_change)
        
        # 切片控制
        ttk.Label(self.toolbar, text="切片:").pack(side=tk.LEFT, padx=(0, 5))
        self.batch_slice_var = tk.StringVar(value="0/0")
        ttk.Label(self.toolbar, textvariable=self.batch_slice_var).pack(side=tk.LEFT, padx=(0, 5))
        
        # 切片导航
        ttk.Button(self.toolbar, text="上一张", command=self.prev_batch_slice).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="下一张", command=self.next_batch_slice).pack(side=tk.LEFT, padx=(0, 5))
        
        # 文件夹选择
        ttk.Button(self.toolbar, text="选择图像文件夹", command=self.open_batch_images_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="选择标签文件夹", command=self.open_batch_labels_folder).pack(side=tk.LEFT, padx=(0, 5))
        
        # 处理控制
        self.process_button = ttk.Button(self.toolbar, text="开始处理", command=self.start_batch_processing)
        self.process_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(self.toolbar, text="停止处理", command=self.stop_batch_processing).pack(side=tk.LEFT, padx=(0, 5))
        
        # 结果操作
        ttk.Button(self.toolbar, text="导出结果", command=self.export_batch_results).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="设置输出目录", command=self.set_output_directory).pack(side=tk.LEFT, padx=(0, 5))
        
        # 调试日志控制
        self.debug_log_var = tk.BooleanVar()
        debug_check = ttk.Checkbutton(self.toolbar, text="调试日志", variable=self.debug_log_var, 
                                    command=self.toggle_debug_log)
        debug_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # 图像信息显示
        self.batch_index_var = tk.StringVar(value="0/0")
        ttk.Label(self.toolbar, textvariable=self.batch_index_var).pack(side=tk.RIGHT, padx=(5, 0))
    
    def toggle_debug_log(self):
        """切换调试日志显示"""
        if hasattr(self, 'debug_log_var'):
            debug_enabled = self.debug_log_var.get()
            if debug_enabled:
                logger.setLevel(logging.DEBUG)
                self.status_var.set("调试日志已启用")
            else:
                logger.setLevel(logging.INFO)
                self.status_var.set("调试日志已禁用")
    
    def handle_mode_specific_keys(self, event):
        """处理批量处理3D模式特定的快捷键"""
        if event.keysym in ["Left", "a", "A"]:
            self.prev_batch_image()
        elif event.keysym in ["Right", "d", "D"]:
            self.next_batch_image()
        elif event.keysym in ["Up", "w", "W"]:
            self.prev_batch_slice()
        elif event.keysym in ["Down", "s", "S"]:
            self.next_batch_slice()
        elif event.keysym == "v":
            # 切换视图
            views = ["axial", "sagittal", "coronal"]
            current_index = views.index(self.batch_current_view)
            next_index = (current_index + 1) % len(views)
            self.batch_view_var.set(views[next_index])
            self.on_batch_view_change()
        elif event.keysym == "space":
            if self.batch_processing:
                self.stop_batch_processing()
            else:
                self.start_batch_processing()
        elif event.state & 0x4 and event.keysym == "e":  # Ctrl+E
            self.export_batch_results()
    
    def on_batch_view_change(self, event=None):
        """批量处理视图改变事件"""
        if not self.batch_3d_viewer:
            return
        
        self.batch_current_view = self.batch_view_var.get()
        self.batch_3d_viewer.set_view(self.batch_current_view)
        
        # 重置切片索引
        view_info = self.batch_3d_viewer.get_view_info(self.batch_current_view)
        if view_info:
            self.batch_current_slice = view_info['max_slices'] // 2
            self.batch_3d_viewer.set_slice(self.batch_current_slice)
        
        # 加载新视图的切片
        self.load_batch_3d_slice()
        
        self.status_var.set(f"切换到{self.batch_current_view}视图")
    
    def prev_batch_slice(self):
        """上一片"""
        if not self.batch_3d_viewer:
            return
        
        view_info = self.batch_3d_viewer.get_view_info(self.batch_current_view)
        if view_info and self.batch_current_slice > 0:
            self.batch_current_slice -= 1
            self.batch_3d_viewer.set_slice(self.batch_current_slice)
            self.load_batch_3d_slice()
    
    def next_batch_slice(self):
        """下一片"""
        if not self.batch_3d_viewer:
            return
        
        view_info = self.batch_3d_viewer.get_view_info(self.batch_current_view)
        if view_info and self.batch_current_slice < view_info['max_slices'] - 1:
            self.batch_current_slice += 1
            self.batch_3d_viewer.set_slice(self.batch_current_slice)
            self.load_batch_3d_slice()
    
    def load_batch_3d_slice(self):
        """加载批量处理3D切片"""
        if not self.batch_3d_viewer:
            return
        
        try:
            # 获取当前切片
            slice_data = self.batch_3d_viewer.get_slice(self.batch_current_view, self.batch_current_slice)
            
            # 转换为8位图像
            if slice_data.dtype != np.uint8:
                slice_data = ((slice_data - slice_data.min()) / 
                            (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            self.ff_image = slice_data
            
            # 更新切片信息显示
            view_info = self.batch_3d_viewer.get_view_info(self.batch_current_view)
            if view_info:
                self.batch_slice_var.set(f"{self.batch_current_slice + 1}/{view_info['max_slices']}")
            
            # 显示图像
            self.display_image()
            self.update_image_info()
            self.update_roi_info()
            
        except Exception as e:
            logger.error(f"加载3D切片失败: {str(e)}")
            self.status_var.set(f"加载切片失败: {str(e)}")
    
    def open_batch_images_folder(self):
        """打开批量图像文件夹"""
        folder_path = filedialog.askdirectory(title="选择图像文件夹")
        if folder_path:
            self.batch_images_dir = folder_path
            self.load_batch_images()
    
    def open_batch_labels_folder(self):
        """打开批量标签文件夹"""
        folder_path = filedialog.askdirectory(title="选择标签文件夹（可选）")
        if folder_path:
            self.batch_labels_dir = folder_path
            self.status_var.set(f"标签文件夹已设置为: {folder_path}")
    
    def load_batch_images(self):
        """加载批量图像列表"""
        if not self.batch_images_dir:
            return
        
        # 获取所有图像文件
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.nii', '.nii.gz')
        self.batch_image_files = []
        
        for file in os.listdir(self.batch_images_dir):
            if file.lower().endswith(image_extensions):
                self.batch_image_files.append(file)
        
        # 使用自然排序
        self.batch_image_files.sort(key=natural_sort_key)
        
        if self.batch_image_files:
            self.batch_current_image_index = 0
            self.show_first_batch_image()
            self.status_var.set(f"已加载 {len(self.batch_image_files)} 个文件")
        else:
            messagebox.showwarning("警告", "文件夹中没有找到图像文件")
    
    def show_first_batch_image(self):
        """显示第一张批量图像"""
        if self.batch_image_files:
            self.batch_current_image_index = 0
            self.show_batch_image_by_index(0)
    
    def start_batch_processing(self):
        """开始批量处理"""
        if not self.batch_image_files:
            messagebox.showwarning("警告", "请先选择图像文件夹")
            return
        
        if not self.output_directory:
            messagebox.showwarning("警告", "请先设置输出目录")
            return
        
        self.batch_processing = True
        self.process_button.config(text="停止处理")
        self.status_var.set("开始批量处理...")
        
        # 在新线程中处理
        import threading
        processing_thread = threading.Thread(target=self.process_batch_images)
        processing_thread.daemon = True
        processing_thread.start()
    
    def process_batch_images(self):
        """处理批量图像"""
        try:
            self.batch_results = []
            
            for i, image_file in enumerate(self.batch_image_files):
                if not self.batch_processing:  # 检查是否停止
                    break
                
                image_path = os.path.join(self.batch_images_dir, image_file)
                
                # 检查是否为3D文件
                if image_file.lower().endswith(('.nii', '.nii.gz')):
                    result = self.process_3d_nifti_batch(image_path, None, image_file)
                else:
                    # 处理2D图像
                    result = self.process_2d_image_batch(image_path, image_file)
                
                if result:
                    self.batch_results.append(result)
                
                # 更新进度
                self.root.after(0, lambda: self.status_var.set(f"处理进度: {i+1}/{len(self.batch_image_files)}"))
            
            # 更新UI
            self.root.after(0, self.batch_processing_completed)
            
        except Exception as e:
            logger.error(f"批量处理失败: {str(e)}")
            self.root.after(0, lambda: self.batch_processing_error(str(e)))
    
    def process_3d_nifti_batch(self, image_path, label_path, image_file):
        """处理3D NIfTI文件"""
        try:
            if not NIBABEL_AVAILABLE:
                logger.warning(f"nibabel库未安装，跳过3D文件: {image_file}")
                return None
            
            # 加载3D文件
            viewer = NIfTI3DViewer(image_path)
            viewer.load_header()
            
            # 获取数据
            image_data = viewer.nii_img.get_fdata()
            
            # 查找对应的标签文件
            if not label_path and self.batch_labels_dir:
                label_path = self.find_corresponding_label(image_file)
            
            label_data = None
            if label_path and os.path.exists(label_path):
                if label_path.lower().endswith(('.nii', '.nii.gz')):
                    # 3D标签文件
                    label_viewer = NIfTI3DViewer(label_path)
                    label_viewer.load_header()
                    label_data = label_viewer.nii_img.get_fdata()
                else:
                    # 2D标签文件
                    label_data = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # 计算脂肪分数
            result = self.calculate_3d_fat_fraction(image_data, label_data, image_file)
            return result
            
        except Exception as e:
            logger.error(f"处理3D文件失败 {image_file}: {str(e)}")
            return None
    
    def process_2d_image_batch(self, image_path, image_file):
        """处理2D图像"""
        try:
            # 加载图像
            image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image_data is None:
                logger.warning(f"无法加载图像: {image_file}")
                return None
            
            # 查找对应的标签文件
            label_path = self.find_corresponding_label(image_file)
            label_data = None
            if label_path and os.path.exists(label_path):
                label_data = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # 计算脂肪分数
            result = self.calculate_2d_fat_fraction(image_data, label_data, image_file)
            return result
            
        except Exception as e:
            logger.error(f"处理2D图像失败 {image_file}: {str(e)}")
            return None
    
    def calculate_3d_fat_fraction(self, image_data, label_data, image_file):
        """计算3D脂肪分数"""
        try:
            # 检测图像像素值范围并决定是否需要归一化
            min_pixel = np.min(image_data)
            max_pixel = np.max(image_data)
            
            # 判断是否需要归一化
            if max_pixel <= 100 and min_pixel >= 0:
                normalized_image = image_data.astype(np.float32)
            else:
                normalized_image = image_data.astype(np.float32) / max_pixel
            
            if label_data is not None:
                # 确保标签数据与图像数据形状一致
                if label_data.shape != image_data.shape:
                    logger.warning(f"标签数据形状不匹配: {image_file}")
                    return None
                
                # 提取ROI区域内的FF值
                roi_binary = (label_data > 0).astype(np.uint8)
                roi_ff_values = normalized_image[roi_binary == 1]
                
                if len(roi_ff_values) == 0:
                    logger.warning(f"ROI区域内没有有效的像素值: {image_file}")
                    return None
                
                # 计算统计信息
                mean_ff = np.mean(roi_ff_values)
                std_ff = np.std(roi_ff_values)
                median_ff = np.median(roi_ff_values)
                min_ff = np.min(roi_ff_values)
                max_ff = np.max(roi_ff_values)
                
                pixel_count = len(roi_ff_values)
                roi_area = np.sum(roi_binary)
                
                result = {
                    'image_file': image_file,
                    'mean_fat_fraction': mean_ff,
                    'std_fat_fraction': std_ff,
                    'median_fat_fraction': median_ff,
                    'min_fat_fraction': min_ff,
                    'max_fat_fraction': max_ff,
                    'pixel_count': pixel_count,
                    'roi_area': roi_area,
                    'roi_coverage_percent': (roi_area / image_data.size) * 100,
                    'original_pixel_range': {'min': int(min_pixel), 'max': int(max_pixel)},
                    'normalization_applied': not (max_pixel <= 100 and min_pixel >= 0),
                    'normalization_factor': max_pixel if not (max_pixel <= 100 and min_pixel >= 0) else 1.0,
                    'data_type': '3d'
                }
                
                return result
            else:
                logger.warning(f"没有找到对应的标签文件: {image_file}")
                return None
                
        except Exception as e:
            logger.error(f"计算3D脂肪分数失败 {image_file}: {str(e)}")
            return None
    
    def calculate_2d_fat_fraction(self, image_data, label_data, image_file):
        """计算2D脂肪分数"""
        try:
            # 检测图像像素值范围并决定是否需要归一化
            min_pixel = np.min(image_data)
            max_pixel = np.max(image_data)
            
            # 判断是否需要归一化
            if max_pixel <= 100 and min_pixel >= 0:
                normalized_image = image_data.astype(np.float32)
            else:
                normalized_image = image_data.astype(np.float32) / max_pixel
            
            if label_data is not None:
                # 确保标签数据与图像数据形状一致
                if label_data.shape != image_data.shape:
                    logger.warning(f"标签数据形状不匹配: {image_file}")
                    return None
                
                # 提取ROI区域内的FF值
                roi_binary = (label_data > 0).astype(np.uint8)
                roi_ff_values = normalized_image[roi_binary == 1]
                
                if len(roi_ff_values) == 0:
                    logger.warning(f"ROI区域内没有有效的像素值: {image_file}")
                    return None
                
                # 计算统计信息
                mean_ff = np.mean(roi_ff_values)
                std_ff = np.std(roi_ff_values)
                median_ff = np.median(roi_ff_values)
                min_ff = np.min(roi_ff_values)
                max_ff = np.max(roi_ff_values)
                
                pixel_count = len(roi_ff_values)
                roi_area = np.sum(roi_binary)
                
                result = {
                    'image_file': image_file,
                    'mean_fat_fraction': mean_ff,
                    'std_fat_fraction': std_ff,
                    'median_fat_fraction': median_ff,
                    'min_fat_fraction': min_ff,
                    'max_fat_fraction': max_ff,
                    'pixel_count': pixel_count,
                    'roi_area': roi_area,
                    'roi_coverage_percent': (roi_area / image_data.size) * 100,
                    'original_pixel_range': {'min': int(min_pixel), 'max': int(max_pixel)},
                    'normalization_applied': not (max_pixel <= 100 and min_pixel >= 0),
                    'normalization_factor': max_pixel if not (max_pixel <= 100 and min_pixel >= 0) else 1.0,
                    'data_type': '2d'
                }
                
                return result
            else:
                logger.warning(f"没有找到对应的标签文件: {image_file}")
                return None
                
        except Exception as e:
            logger.error(f"计算2D脂肪分数失败 {image_file}: {str(e)}")
            return None
    
    def batch_processing_completed(self):
        """批量处理完成"""
        self.batch_processing = False
        self.process_button.config(text="开始处理")
        self.status_var.set(f"批量处理完成，共处理 {len(self.batch_results)} 个文件")
        messagebox.showinfo("完成", f"批量处理完成，共处理 {len(self.batch_results)} 个文件")
    
    def batch_processing_error(self, error_msg):
        """批量处理错误"""
        self.batch_processing = False
        self.process_button.config(text="开始处理")
        self.status_var.set(f"批量处理失败: {error_msg}")
        messagebox.showerror("错误", f"批量处理失败: {error_msg}")
    
    def stop_batch_processing(self):
        """停止批量处理"""
        self.batch_processing = False
        self.process_button.config(text="开始处理")
        self.status_var.set("批量处理已停止")
    
    def prev_batch_image(self):
        """上一张批量图像"""
        if not self.batch_image_files:
            return
        
        self.batch_current_image_index = (self.batch_current_image_index - 1) % len(self.batch_image_files)
        self.show_batch_image_by_index(self.batch_current_image_index)
    
    def next_batch_image(self):
        """下一张批量图像"""
        if not self.batch_image_files:
            return
        
        self.batch_current_image_index = (self.batch_current_image_index + 1) % len(self.batch_image_files)
        self.show_batch_image_by_index(self.batch_current_image_index)
    
    def show_batch_image_by_index(self, index):
        """显示指定索引的批量图像"""
        if not self.batch_image_files or index >= len(self.batch_image_files):
            return
        
        try:
            # 加载图像
            image_file = self.batch_image_files[index]
            image_path = os.path.join(self.batch_images_dir, image_file)
            
            # 检查是否为3D文件
            if image_file.lower().endswith(('.nii', '.nii.gz')):
                if not NIBABEL_AVAILABLE:
                    messagebox.showerror("错误", "nibabel库未安装，无法处理3D文件")
                    return
                
                # 加载3D文件
                self.batch_3d_viewer = NIfTI3DViewer(image_path)
                self.batch_3d_viewer.load_header()
                
                # 设置默认视图和切片
                self.batch_current_view = 'axial'
                self.batch_current_slice = self.batch_3d_viewer.current_slice
                
                # 加载当前切片
                self.load_batch_3d_slice()
                
            else:
                # 加载2D图像
                self.ff_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if self.ff_image is None:
                    raise ValueError(f"无法加载图像: {image_path}")
                
                self.batch_3d_viewer = None
                
                # 显示图像
                self.display_image()
                self.update_image_info()
                self.update_roi_info()
            
            # 查找对应的标签文件
            label_path = self.find_corresponding_label(image_file)
            if label_path:
                if label_path.lower().endswith(('.nii', '.nii.gz')):
                    # 3D标签文件
                    label_viewer = NIfTI3DViewer(label_path)
                    label_viewer.load_header()
                    # 这里可以添加3D标签处理逻辑
                else:
                    # 2D标签文件
                    self.roi_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    if self.roi_mask is not None:
                        # 将掩码转换为ROI列表
                        self.convert_mask_to_roi_list(self.roi_mask)
                    else:
                        self.roi_list = []
            else:
                self.roi_list = []
                self.roi_mask = None
            
            # 重置旋转角度
            self.rotation_angle = 0
            
            # 更新显示
            self.update_batch_index()
            
            # 显示处理结果（如果有）
            if index < len(self.batch_results):
                self.display_batch_result(index)
            
            self.status_var.set(f"显示文件: {image_file}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")
            logger.error(f"加载文件失败: {str(e)}")
    
    def display_batch_result(self, index):
        """显示批量处理结果"""
        if index >= len(self.batch_results):
            return
        
        result = self.batch_results[index]
        
        # 显示结果信息
        if 'mean_fat_fraction' in result:
            self.status_var.set(f"平均脂肪分数: {result['mean_fat_fraction']:.3f}")
        
        # 更新ROI信息
        if 'roi_area' in result:
            self.roi_info_var.set(f"ROI面积: {result['roi_area']} 像素")
    
    def update_batch_index(self):
        """更新批量处理索引显示"""
        if self.batch_image_files:
            self.batch_index_var.set(f"{self.batch_current_image_index + 1}/{len(self.batch_image_files)}")
        else:
            self.batch_index_var.set("0/0")
    
    def export_batch_results(self):
        """导出批量处理结果"""
        if not self.batch_results:
            messagebox.showwarning("警告", "没有结果可导出")
            return
        
        if not self.output_directory:
            messagebox.showwarning("警告", "请先设置输出目录")
            return
        
        try:
            # 创建汇总表格
            import pandas as pd
            
            summary_data = []
            for result in self.batch_results:
                summary_data.append({
                    '文件': result.get('image_file', 'unknown'),
                    '数据类型': result.get('data_type', 'unknown'),
                    '平均脂肪分数': result.get('mean_fat_fraction', 0),
                    '标准差': result.get('std_fat_fraction', 0),
                    '中位数': result.get('median_fat_fraction', 0),
                    '像素数量': result.get('pixel_count', 0),
                    'ROI面积': result.get('roi_area', 0),
                    '覆盖率(%)': result.get('roi_coverage_percent', 0)
                })
            
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.output_directory, "batch_3d_results_summary.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # 保存JSON结果
            import json
            json_path = os.path.join(self.output_directory, "batch_3d_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.batch_results, f, indent=2, ensure_ascii=False)
            
            self.status_var.set(f"结果已导出到: {self.output_directory}")
            messagebox.showinfo("成功", f"结果已导出到: {self.output_directory}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")
            logger.error(f"导出失败: {str(e)}")
