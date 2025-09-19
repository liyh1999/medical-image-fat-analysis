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
        # 先定义进度条相关变量（在调用父类初始化之前）
        self.progress_var = tk.DoubleVar()
        self.progress_label_var = tk.StringVar()
        
        super().__init__(parent)
        
        # 批量处理3D控制变量
        self.batch_3d_viewer = None  # 批量处理3D查看器
        self.batch_current_view = 'axial'  # 批量处理当前视图
        self.batch_current_slice = 0  # 批量处理当前切片
        self.batch_current_image_index = 0  # 当前显示的图像索引
        
        # 批量处理相关
        self.batch_images_dir = None
        self.batch_labels_dir = None
        self.batch_image_files = []
        self.batch_results = []
        self.batch_processing = False
        
        # 缓存相关（用于优化性能）
        self._current_label_file = None
        self._label_viewer = None
        self._label_path = None
        self._roi_cache = {}  # 缓存ROI数据，key为(view, slice)
        
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
            
            # 应用旋转
            if self.rotation_angle != 0:
                rotated_image = self.apply_rotation(slice_data, self.rotation_angle)
            else:
                rotated_image = slice_data.copy()
            
            # 设置FF图像数据（用于ROI计算）
            self.ff_image = slice_data.copy()  # 使用原始图像数据计算ROI
            
            # 绘制ROI到旋转后的图像上（类似2D批量处理）
            display_image = rotated_image.copy()
            if hasattr(self, 'roi_list') and self.roi_list:
                display_image = self.draw_rois_on_image(display_image)
            
            # 转换为PIL图像
            if len(display_image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(display_image)
            
            # 计算缩放比例
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                scale_x = canvas_width / display_image.shape[1]
                scale_y = canvas_height / display_image.shape[0]
                self.scale_factor = min(scale_x, scale_y, 1.0)
                
                # 计算图像在画布中的位置
                new_width = int(display_image.shape[1] * self.scale_factor)
                new_height = int(display_image.shape[0] * self.scale_factor)
                
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
            
        except Exception as e:
            logger.error(f"显示批量处理3D切片失败: {str(e)}")
            if hasattr(self, 'status_var') and self.status_var:
                self.status_var.set(f"显示切片失败: {str(e)}")
        
    def create_toolbar(self, parent):
        """创建批量处理3D模式工具栏（只展示和读取标注，不勾画）"""
        # 创建工具栏框架
        self.toolbar = ttk.Frame(parent)
        self.toolbar.pack(fill=tk.X, padx=5, pady=2)
        
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
        
        # 文件导航
        ttk.Button(self.toolbar, text="上一个文件", command=self.prev_batch_image).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(self.toolbar, text="下一个文件", command=self.next_batch_image).pack(side=tk.LEFT, padx=(0, 5))
        
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
        
        # 进度条
        self.progress_bar = ttk.Progressbar(self.toolbar, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        # 进度标签
        self.progress_label = ttk.Label(self.toolbar, textvariable=self.progress_label_var, font=("Arial", 9))
        self.progress_label.pack(side=tk.LEFT, padx=(5, 0))
        
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
    
    def clear_viewer_cache(self):
        """清理3D查看器缓存"""
        if self.batch_3d_viewer:
            self.batch_3d_viewer.clear_cache()
            logger.info("已清理3D查看器缓存")
        
        # 清理ROI缓存
        self._roi_cache.clear()
        self._current_label_file = None
        self._label_viewer = None
        self._label_path = None
        # 不要重置 batch_3d_viewer，只是清理缓存
    
    def handle_mode_specific_keys(self, event):
        """处理批量处理3D模式特定的快捷键（只展示和读取标注）"""
        if event.keysym in ["Left", "a", "A"]:
            self.prev_batch_slice()  # A/D键切换层面
        elif event.keysym in ["Right", "d", "D"]:
            self.next_batch_slice()  # A/D键切换层面
        elif event.keysym in ["Up", "w", "W"]:
            self.prev_batch_image()  # W/S键切换文件
        elif event.keysym in ["Down", "s", "S"]:
            self.next_batch_image()  # W/S键切换文件
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
    
    def on_mouse_click(self, event):
        """鼠标点击事件（批量3D模式 - 禁用勾画功能）"""
        # 批量3D模式只展示和读取标注，不进行勾画
        pass
    
    def on_mouse_drag(self, event):
        """鼠标拖拽事件（批量3D模式 - 禁用勾画功能）"""
        # 批量3D模式只展示和读取标注，不进行勾画
        pass
    
    def on_mouse_release(self, event):
        """鼠标释放事件（批量3D模式 - 禁用勾画功能）"""
        # 批量3D模式只展示和读取标注，不进行勾画
        pass
    
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
        """加载批量处理3D切片（优化版）"""
        if not self.batch_3d_viewer:
            return
        
        try:
            # 获取当前切片
            slice_data = self.batch_3d_viewer.get_slice(self.batch_current_view, self.batch_current_slice)
            
            # 转换为8位图像（优化：只在需要时转换）
            if slice_data.dtype != np.uint8:
                slice_data = ((slice_data - slice_data.min()) / 
                            (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            self.ff_image = slice_data
            
            # 更新切片信息显示
            view_info = self.batch_3d_viewer.get_view_info(self.batch_current_view)
            if view_info:
                self.batch_slice_var.set(f"{self.batch_current_slice + 1}/{view_info['max_slices']}")
            
            # 重新加载当前切片的标签数据（使用原始图像数据）
            self.reload_current_slice_labels()
            
            # 显示图像（在ROI计算之后）
            self.display_image()
            
            # 合并更新调用，减少重复计算
            self.update_image_info()
            # update_roi_info 已经在 reload_current_slice_labels 中调用了
            
        except Exception as e:
            logger.error(f"加载3D切片失败: {str(e)}")
            self.status_var.set(f"加载切片失败: {str(e)}")
    
    def convert_3d_label_slice_to_roi_list(self, label_slice):
        """将3D标签切片转换为ROI列表，支持多标签"""
        roi_list = []
        
        # 获取所有唯一的标签值（排除0）
        unique_labels = np.unique(label_slice)
        unique_labels = unique_labels[unique_labels > 0]
        
        logger.debug(f"发现 {len(unique_labels)} 个标签值: {unique_labels}")
        
        total_labels = len(unique_labels)
        for idx, label_value in enumerate(unique_labels):
            # 更新进度条（只在批量处理时显示）
            if hasattr(self, 'batch_processing') and self.batch_processing and total_labels > 1:
                progress = (idx / total_labels) * 100
                self.progress_var.set(progress)
                self.progress_label_var.set(f"处理标签 {label_value} ({idx+1}/{total_labels})")
                self.root.update()  # 更新界面
            
            # 为每个标签值创建掩码
            mask = (label_slice == label_value).astype(np.uint8) * 255
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area < 100:  # 过滤面积小于100像素的区域
                    continue
                
                # 计算FF值（优化：只在需要时计算）
                fat_fraction = None
                if hasattr(self, 'ff_image') and self.ff_image is not None:
                    fat_fraction = self.calculate_roi_fat_fraction(contour, label_slice)
                
                # 创建多边形ROI（保持原始坐标，旋转在显示时处理）
                roi = {
                    'type': 'polygon',
                    'points': contour.reshape(-1, 2).tolist(),
                    'fat_fraction': fat_fraction,
                    'label_value': int(label_value)  # 添加标签值信息
                }
                roi_list.append(roi)
        
        # 完成进度条（只在批量处理时显示）
        if hasattr(self, 'batch_processing') and self.batch_processing and total_labels > 1:
            self.progress_var.set(100)
            self.progress_label_var.set("完成")
            self.root.update()
        else:
            # 隐藏进度条
            self.progress_var.set(0)
            self.progress_label_var.set("")
        
        logger.info(f"从3D标签切片转换得到 {len(roi_list)} 个ROI")
        return roi_list
    
    def calculate_label_statistics(self, roi_list):
        """计算每个标签的统计信息"""
        label_stats = {}
        
        for roi in roi_list:
            if 'label_value' not in roi or roi['fat_fraction'] is None:
                continue
                
            label_value = roi['label_value']
            if label_value not in label_stats:
                label_stats[label_value] = {
                    'fat_fractions': [],
                    'pixel_counts': [],
                    'roi_count': 0
                }
            
            # 收集数据
            if 'fat_fraction' in roi['fat_fraction']:
                label_stats[label_value]['fat_fractions'].append(roi['fat_fraction']['fat_fraction'])
            if 'pixel_count' in roi['fat_fraction']:
                label_stats[label_value]['pixel_counts'].append(roi['fat_fraction']['pixel_count'])
            label_stats[label_value]['roi_count'] += 1
        
        # 计算统计值
        for label_value, stats in label_stats.items():
            if stats['fat_fractions']:
                stats['mean_fat_fraction'] = np.mean(stats['fat_fractions'])
                stats['std_fat_fraction'] = np.std(stats['fat_fractions'])
                stats['median_fat_fraction'] = np.median(stats['fat_fractions'])
            else:
                stats['mean_fat_fraction'] = 0
                stats['std_fat_fraction'] = 0
                stats['median_fat_fraction'] = 0
                
            stats['pixel_count'] = sum(stats['pixel_counts']) if stats['pixel_counts'] else 0
            
            # 清理临时数据
            del stats['fat_fractions']
            del stats['pixel_counts']
        
        return label_stats
    
    def calculate_3d_label_statistics(self, label_data, normalized_image):
        """直接从3D标签数据计算标签统计信息"""
        label_stats = {}
        
        # 获取所有唯一的标签值（排除0）
        unique_labels = np.unique(label_data)
        unique_labels = unique_labels[unique_labels > 0]
        
        logger.info(f"发现 {len(unique_labels)} 个标签值: {unique_labels}")
        
        for label_value in unique_labels:
            # 创建当前标签的掩码
            label_mask = (label_data == label_value)
            
            # 提取该标签对应的像素值
            label_pixels = normalized_image[label_mask]
            
            if len(label_pixels) > 0:
                # 计算统计信息
                mean_ff = np.mean(label_pixels)
                std_ff = np.std(label_pixels)
                median_ff = np.median(label_pixels)
                pixel_count = len(label_pixels)
                
                # 对于3D数据，简化ROI计数（避免性能问题）
                # 使用标签值的出现次数作为ROI数量的估算
                roi_count = 1  # 简化处理，每个标签值算作1个ROI
                
                label_stats[label_value] = {
                    'mean_fat_fraction': float(mean_ff),
                    'std_fat_fraction': float(std_ff),
                    'median_fat_fraction': float(median_ff),
                    'pixel_count': int(pixel_count),
                    'roi_count': int(roi_count)
                }
                
                logger.info(f"标签 {label_value}: FF={mean_ff:.3f}, 像素数={pixel_count:,}")
        
        return label_stats
    
    def count_connected_components(self, mask):
        """计算连通区域数量"""
        try:
            import cv2
            # 确保掩码是单通道uint8类型
            if len(mask.shape) > 2:
                mask_uint8 = mask[:, :, 0].astype(np.uint8)
            else:
                mask_uint8 = mask.astype(np.uint8)
            
            # 确保是单通道
            if len(mask_uint8.shape) != 2:
                logger.warning(f"掩码形状不正确: {mask_uint8.shape}")
                return 1
                
            # 找到连通区域
            num_labels, _ = cv2.connectedComponents(mask_uint8)
            # 减去背景（标签0）
            return max(0, num_labels - 1)
        except Exception as e:
            logger.warning(f"计算连通区域失败: {str(e)}")
            # 使用简单的估算方法
            return 1
    
    def calculate_roi_fat_fraction(self, contour, label_slice):
        """计算ROI的脂肪分数"""
        if self.ff_image is None:
            return None
        
        try:
            # 确保轮廓是正确格式
            if len(contour.shape) == 3:
                contour = contour.reshape(-1, 2)
            
            # 创建掩码
            mask = np.zeros(self.ff_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
            
            # 获取ROI区域内的像素值
            roi_pixels = self.ff_image[mask > 0]
            
            if len(roi_pixels) == 0:
                return None
            
            # 计算脂肪分数（这里使用简单的统计方法，您可以根据需要调整）
            mean_value = np.mean(roi_pixels)
            std_value = np.std(roi_pixels)
            min_value = np.min(roi_pixels)
            max_value = np.max(roi_pixels)
            
            # 改进的脂肪分数计算
            # 使用更合理的FF计算公式
            if max_value > 0:
                fat_fraction = mean_value / max_value
            else:
                fat_fraction = 0
            
            # 添加调试信息
            logger.debug(f"ROI像素统计: 均值={mean_value:.2f}, 标准差={std_value:.2f}, 最小值={min_value}, 最大值={max_value}, FF={fat_fraction:.3f}")
            
            return {
                'mean': float(mean_value),
                'std': float(std_value),
                'min': float(min_value),
                'max': float(max_value),
                'fat_fraction': float(fat_fraction),
                'pixel_count': len(roi_pixels)
            }
        except Exception as e:
            logger.error(f"计算ROI脂肪分数失败: {str(e)}")
            return None
    
    def reload_current_slice_labels(self):
        """重新加载当前切片的标签数据（优化版）"""
        if not hasattr(self, 'batch_image_files') or not self.batch_image_files:
            return
        
        if not hasattr(self, 'batch_current_image_index'):
            return
        
        # 检查缓存
        cache_key = (self.batch_current_view, self.batch_current_slice)
        if cache_key in self._roi_cache:
            self.roi_list = self._roi_cache[cache_key].copy()
            self.update_roi_info()
            return
        
        # 获取当前图像文件
        current_image_file = self.batch_image_files[self.batch_current_image_index]
        
        # 检查是否需要重新加载标签文件
        if self._current_label_file != current_image_file:
            self._current_label_file = current_image_file
            self._label_viewer = None
            self._label_path = None
            self._roi_cache.clear()  # 清空缓存
        
        # 获取标签文件路径（缓存）
        if not self._label_path:
            self._label_path = self.find_corresponding_label(current_image_file)
        
        if self._label_path and self._label_path.lower().endswith(('.nii', '.nii.gz')):
            try:
                # 使用缓存的标签viewer
                if not self._label_viewer:
                    self._label_viewer = NIfTI3DViewer(self._label_path)
                    self._label_viewer.load_header()
                
                # 获取当前切片的标签数据
                label_slice = self._label_viewer.get_slice(self.batch_current_view, self.batch_current_slice)
                if label_slice is not None:
                    # 使用当前的ff_image，避免重复转换
                    if hasattr(self, 'ff_image') and self.ff_image is not None:
                        # 处理多标签情况
                        self.roi_list = self.convert_3d_label_slice_to_roi_list(label_slice)
                        logger.debug(f"重新加载切片标签: {len(self.roi_list)} 个ROI")
                        
                        # 缓存ROI数据
                        self._roi_cache[cache_key] = [roi.copy() for roi in self.roi_list]
                        
                        # 更新ROI信息显示
                        self.update_roi_info()
                    else:
                        logger.warning("无法获取图像数据")
                        self.roi_list = []
                        self.roi_mask = None
                        self.update_roi_info()
                else:
                    self.roi_list = []
                    self.roi_mask = None
                    self.update_roi_info()
            except Exception as e:
                logger.error(f"重新加载切片标签失败: {str(e)}")
                self.roi_list = []
                self.roi_mask = None
                self.update_roi_info()
        else:
            # 没有标签文件或不是3D标签文件
            self.roi_list = []
            self.roi_mask = None
            self.update_roi_info()
    
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
            
            # 重新加载当前图像的标签
            if hasattr(self, 'batch_image_files') and self.batch_image_files:
                self.show_batch_image_by_index(self.batch_current_image_index)
    
    def load_batch_images(self):
        """加载批量图像列表"""
        if not self.batch_images_dir:
            return
        
        # 获取所有图像文件
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.nii', '.nii.gz')
        self.batch_image_files = []
        
        for file in os.listdir(self.batch_images_dir):
            # 检查文件扩展名（不区分大小写）
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in image_extensions):
                self.batch_image_files.append(file)
                logger.debug(f"找到图像文件: {file}")
        
        # 使用自然排序
        self.batch_image_files.sort(key=natural_sort_key)
        logger.info(f"共找到 {len(self.batch_image_files)} 个图像文件: {self.batch_image_files}")
        
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
        
        # 启动进度条动画定时器
        self.start_progress_animation()
        
        # 在新线程中处理
        import threading
        processing_thread = threading.Thread(target=self.process_batch_images)
        processing_thread.daemon = True
        processing_thread.start()
    
    def start_progress_animation(self):
        """启动进度条动画，让进度条慢慢增长"""
        self.progress_animation_running = True
        self.animate_progress()
    
    def animate_progress(self):
        """进度条动画"""
        if not self.batch_processing or not self.progress_animation_running:
            return
        
        # 获取当前进度
        current_progress = self.progress_var.get()
        
        # 如果进度条没有在增长，让它慢慢增长（欺骗用户）
        if current_progress < 100:
            # 每次增长0.5%，让进度条慢慢移动
            new_progress = min(current_progress + 0.5, 100)
            self.progress_var.set(new_progress)
            
            # 更新状态信息
            if hasattr(self, 'current_file_index') and hasattr(self, 'batch_image_files'):
                current_file = self.current_file_index + 1
                total_files = len(self.batch_image_files)
                self.progress_label_var.set(f"处理文件: {current_file}/{total_files}")
                self.status_var.set(f"处理进度: {current_file}/{total_files} - 处理中...")
        
        # 每100毫秒更新一次
        self.root.after(100, self.animate_progress)
    
    def stop_progress_animation(self):
        """停止进度条动画"""
        self.progress_animation_running = False
    
    def process_batch_images(self):
        """处理批量图像"""
        try:
            self.batch_results = []
            self.current_file_index = 0
            self.file_progress = 0  # 当前文件内部进度 (0-100)
            
            for i, image_file in enumerate(self.batch_image_files):
                if not self.batch_processing:  # 检查是否停止
                    break
                
                self.current_file_index = i
                self.file_progress = 0
                
                # 开始处理文件时更新进度
                self.root.after(0, lambda idx=i+1, total=len(self.batch_image_files): 
                              self.update_batch_progress(0, idx, total, "开始处理"))
                
                image_path = os.path.join(self.batch_images_dir, image_file)
                
                # 检查是否为3D文件
                if image_file.lower().endswith(('.nii', '.nii.gz')):
                    result = self.process_3d_nifti_batch(image_path, None, image_file)
                else:
                    # 处理2D图像
                    result = self.process_2d_image_batch(image_path, image_file)
                
                if result:
                    self.batch_results.append(result)
                
                # 文件处理完成，更新进度
                progress = ((i + 1) / len(self.batch_image_files)) * 100
                self.root.after(0, lambda p=progress, idx=i+1, total=len(self.batch_image_files): 
                              self.update_batch_progress(p, idx, total, "完成"))
            
            # 更新UI
            self.root.after(0, self.batch_processing_completed)
            
        except Exception as e:
            logger.error(f"批量处理失败: {str(e)}")
            self.root.after(0, lambda: self.batch_processing_error(str(e)))
    
    def update_batch_progress(self, progress, current, total, status=""):
        """更新批量处理进度"""
        self.progress_var.set(progress)
        self.progress_label_var.set(f"处理文件: {current}/{total}")
        if status:
            self.status_var.set(f"处理进度: {current}/{total} - {status}")
        else:
            self.status_var.set(f"处理进度: {current}/{total}")
    
    def update_file_progress(self, file_progress, status=""):
        """更新当前文件内部进度"""
        if hasattr(self, 'current_file_index') and hasattr(self, 'batch_image_files'):
            # 计算总体进度：已完成文件 + 当前文件进度
            completed_files = self.current_file_index
            total_files = len(self.batch_image_files)
            current_file_weight = 1.0 / total_files  # 当前文件占总进度的权重
            
            # 总体进度 = (已完成文件数 + 当前文件进度) / 总文件数 * 100
            overall_progress = (completed_files + (file_progress / 100.0)) / total_files * 100
            
            self.progress_var.set(overall_progress)
            self.progress_label_var.set(f"处理文件: {self.current_file_index + 1}/{total_files}")
            if status:
                self.status_var.set(f"处理进度: {self.current_file_index + 1}/{total_files} - {status}")
            else:
                self.status_var.set(f"处理进度: {self.current_file_index + 1}/{total_files}")
    
    def process_3d_nifti_batch(self, image_path, label_path, image_file):
        """处理3D NIfTI文件"""
        try:
            if not NIBABEL_AVAILABLE:
                logger.warning(f"nibabel库未安装，跳过3D文件: {image_file}")
                return None
            
            logger.info(f"开始处理3D文件: {image_file}")
            
            # 更新进度：加载图像
            self.root.after(0, lambda: self.update_file_progress(20, "加载图像"))
            
            # 直接使用nibabel加载，避免NIfTI3DViewer的缓存开销
            import nibabel as nib
            nii_img = nib.load(image_path)
            image_data = nii_img.get_fdata()
            
            logger.info(f"图像数据形状: {image_data.shape}, 数据类型: {image_data.dtype}")
            
            # 更新进度：查找标签
            self.root.after(0, lambda: self.update_file_progress(40, "查找标签文件"))
            
            # 查找对应的标签文件
            if not label_path and self.batch_labels_dir:
                label_path = self.find_corresponding_label(image_file)
            
            label_data = None
            if label_path and os.path.exists(label_path):
                logger.info(f"找到标签文件: {label_path}")
                if label_path.lower().endswith(('.nii', '.nii.gz')):
                    # 3D标签文件
                    label_nii = nib.load(label_path)
                    label_data = label_nii.get_fdata()
                    logger.info(f"标签数据形状: {label_data.shape}, 数据类型: {label_data.dtype}")
                else:
                    # 2D标签文件
                    label_data = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            else:
                logger.warning(f"未找到标签文件: {image_file}")
            
            # 更新进度：计算脂肪分数
            self.root.after(0, lambda: self.update_file_progress(60, "计算脂肪分数"))
            
            # 计算脂肪分数
            logger.info(f"开始计算脂肪分数: {image_file}")
            result = self.calculate_3d_fat_fraction(image_data, label_data, image_file)
            
            # 更新进度：完成
            self.root.after(0, lambda: self.update_file_progress(100, "完成"))
            
            if result:
                logger.info(f"完成处理: {image_file}")
            else:
                logger.warning(f"处理失败: {image_file}")
            
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
            # 更新进度：检测像素值范围
            self.root.after(0, lambda: self.update_file_progress(65, "检测像素值范围"))
            
            # 检测图像像素值范围并决定是否需要归一化
            min_pixel = np.min(image_data)
            max_pixel = np.max(image_data)
            
            # 判断是否需要归一化
            if max_pixel <= 100 and min_pixel >= 0:
                normalized_image = image_data.astype(np.float32)
            else:
                normalized_image = image_data.astype(np.float32) / max_pixel
            
            # 更新进度：处理标签数据
            self.root.after(0, lambda: self.update_file_progress(75, "处理标签数据"))
            
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
                
                # 更新进度：计算统计信息
                self.root.after(0, lambda: self.update_file_progress(85, "计算统计信息"))
                
                # 计算统计信息
                mean_ff = np.mean(roi_ff_values)
                std_ff = np.std(roi_ff_values)
                median_ff = np.median(roi_ff_values)
                min_ff = np.min(roi_ff_values)
                max_ff = np.max(roi_ff_values)
                
                pixel_count = len(roi_ff_values)
                roi_area = np.sum(roi_binary)
                
                # 更新进度：计算标签统计
                self.root.after(0, lambda: self.update_file_progress(90, "计算标签统计"))
                
                # 计算标签统计信息
                label_stats = self.calculate_3d_label_statistics(label_data, normalized_image)
                
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
                    'data_type': '3d',
                    'label_stats': label_stats
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
        self.stop_progress_animation()
        self.process_button.config(text="开始处理")
        self.progress_var.set(100)
        self.progress_label_var.set("处理完成")
        self.status_var.set(f"批量处理完成，共处理 {len(self.batch_results)} 个文件")
        messagebox.showinfo("完成", f"批量处理完成，共处理 {len(self.batch_results)} 个文件")
    
    def batch_processing_error(self, error_msg):
        """批量处理错误"""
        self.batch_processing = False
        self.stop_progress_animation()
        self.process_button.config(text="开始处理")
        self.progress_var.set(0)
        self.progress_label_var.set("")
        self.status_var.set(f"批量处理失败: {error_msg}")
        messagebox.showerror("错误", f"批量处理失败: {error_msg}")
    
    def stop_batch_processing(self):
        """停止批量处理"""
        self.batch_processing = False
        self.stop_progress_animation()
        self.process_button.config(text="开始处理")
        self.status_var.set("批量处理已停止")
        self.progress_var.set(0)
        self.progress_label_var.set("")
    
    def prev_batch_image(self):
        """上一张批量图像"""
        if not self.batch_image_files:
            return
        
        # 清理当前文件的缓存
        self.clear_viewer_cache()
        
        self.batch_current_image_index = (self.batch_current_image_index - 1) % len(self.batch_image_files)
        self.show_batch_image_by_index(self.batch_current_image_index)
    
    def next_batch_image(self):
        """下一张批量图像"""
        if not self.batch_image_files:
            return
        
        # 清理当前文件的缓存
        self.clear_viewer_cache()
        
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
                
                # 检查是否需要重新加载3D文件（避免重复加载相同文件）
                if (self.batch_3d_viewer is None or 
                    self.batch_3d_viewer.file_path != image_path):
                    # 清理前一个文件的缓存
                    if self.batch_3d_viewer:
                        self.batch_3d_viewer.clear_cache()
                    
                    # 加载新的3D文件
                    self.batch_3d_viewer = NIfTI3DViewer(image_path)
                    self.batch_3d_viewer.load_header()
                    logger.info(f"加载新的3D文件: {image_file}")
                else:
                    logger.info(f"重用已加载的3D文件: {image_file}")
                
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
                    try:
                        label_viewer = NIfTI3DViewer(label_path)
                        label_viewer.load_header()
                        
                        # 获取当前切片的标签数据
                        if self.batch_3d_viewer:
                            label_slice = label_viewer.get_slice(self.batch_current_view, self.batch_current_slice)
                            if label_slice is not None:
                                # 处理多标签情况
                                self.roi_list = self.convert_3d_label_slice_to_roi_list(label_slice)
                                logger.info(f"从3D标签文件加载了 {len(self.roi_list)} 个ROI")
                                
                                # 计算标签统计信息
                                if hasattr(self, 'batch_results') and self.batch_current_image_index < len(self.batch_results):
                                    label_stats = self.calculate_label_statistics(self.roi_list)
                                    self.batch_results[self.batch_current_image_index]['label_stats'] = label_stats
                            else:
                                self.roi_list = []
                                self.roi_mask = None
                        else:
                            self.roi_list = []
                            self.roi_mask = None
                    except Exception as e:
                        logger.error(f"加载3D标签文件失败: {str(e)}")
                        self.roi_list = []
                        self.roi_mask = None
                else:
                    # 2D标签文件
                    self.roi_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    if self.roi_mask is not None:
                        # 将掩码转换为ROI列表
                        self.roi_list = self.convert_mask_to_roi_list(self.roi_mask)
                        logger.info(f"从2D标签文件加载了 {len(self.roi_list)} 个ROI")
                    else:
                        self.roi_list = []
            else:
                self.roi_list = []
                self.roi_mask = None
                logger.info(f"未找到对应的标签文件: {image_file}")
                logger.info(f"标签文件夹: {self.batch_labels_dir}")
                if self.batch_labels_dir and os.path.exists(self.batch_labels_dir):
                    logger.info(f"标签文件夹存在，包含文件: {os.listdir(self.batch_labels_dir)}")
                else:
                    logger.info(f"标签文件夹不存在或未设置")
            
            # 保持旋转角度（不重置）
            # self.rotation_angle = 0  # 注释掉，保持当前旋转角度
            
            # 重新显示图像（包含标签）
            self.display_image()
            
            # 更新显示
            self.update_batch_index()
            
            # 显示处理结果（如果有）
            if hasattr(self, 'batch_results') and index < len(self.batch_results):
                self.display_batch_result(index)
            
            # 更新状态显示
            if self.batch_3d_viewer:
                view_info = self.batch_3d_viewer.get_view_info(self.batch_current_view)
                if view_info:
                    self.status_var.set(f"显示文件: {image_file} | 视图: {self.batch_current_view} | 切片: {self.batch_current_slice + 1}/{view_info['max_slices']}")
                else:
                    self.status_var.set(f"显示文件: {image_file} | 视图: {self.batch_current_view}")
            else:
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
            import pandas as pd
            import json
            
            # 1. 创建文件汇总表格
            file_summary_data = []
            for result in self.batch_results:
                file_info = {
                    '文件': result.get('image_file', 'unknown'),
                    '数据类型': result.get('data_type', 'unknown'),
                    '总平均脂肪分数': result.get('mean_fat_fraction', 0),
                    '总标准差': result.get('std_fat_fraction', 0),
                    '总中位数': result.get('median_fat_fraction', 0),
                    '总像素数量': result.get('pixel_count', 0),
                    '总ROI面积': result.get('roi_area', 0),
                    '总覆盖率(%)': result.get('roi_coverage_percent', 0)
                }
                file_summary_data.append(file_info)
            
            # 保存文件汇总
            df_files = pd.DataFrame(file_summary_data)
            csv_files_path = os.path.join(self.output_directory, "batch_3d_files_summary.csv")
            df_files.to_csv(csv_files_path, index=False, encoding='utf-8-sig')
            
            # 2. 创建ROI详细统计表格
            roi_detail_data = []
            for result in self.batch_results:
                image_file = result.get('image_file', 'unknown')
                
                # 添加每个标签的ROI统计
                if 'label_stats' in result:
                    for label_value, stats in result['label_stats'].items():
                        roi_info = {
                            '文件': image_file,
                            '标签值': label_value,
                            '平均脂肪分数': stats.get('mean_fat_fraction', 0),
                            '标准差': stats.get('std_fat_fraction', 0),
                            '中位数': stats.get('median_fat_fraction', 0),
                            '像素数量': stats.get('pixel_count', 0),
                            'ROI数量': stats.get('roi_count', 0)
                        }
                        roi_detail_data.append(roi_info)
            
            # 保存ROI详细统计
            if roi_detail_data:
                df_rois = pd.DataFrame(roi_detail_data)
                csv_rois_path = os.path.join(self.output_directory, "batch_3d_rois_detail.csv")
                df_rois.to_csv(csv_rois_path, index=False, encoding='utf-8-sig')
            
            # 3. 保存JSON结果（包含所有详细信息）
            def convert_numpy_types(obj):
                """递归转换NumPy类型为Python原生类型"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # 转换数据
            converted_results = convert_numpy_types(self.batch_results)
            
            json_path = os.path.join(self.output_directory, "batch_3d_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
            # 4. 创建统计报告
            self.create_statistics_report()
            
            self.status_var.set(f"结果已导出到: {self.output_directory}")
            messagebox.showinfo("成功", f"结果已导出到: {self.output_directory}\n\n包含文件:\n- batch_3d_files_summary.csv (文件汇总)\n- batch_3d_rois_detail.csv (ROI详细统计)\n- batch_3d_results.json (完整数据)\n- batch_3d_statistics_report.txt (统计报告)")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")
            logger.error(f"导出失败: {str(e)}")
    
    def create_statistics_report(self):
        """创建统计报告"""
        try:
            report_path = os.path.join(self.output_directory, "batch_3d_statistics_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("3D批量处理统计报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 总体统计
                f.write("总体统计:\n")
                f.write(f"处理文件数量: {len(self.batch_results)}\n")
                
                total_rois = 0
                total_pixels = 0
                all_ff_values = []
                
                for result in self.batch_results:
                    if 'label_stats' in result:
                        for label_value, stats in result['label_stats'].items():
                            total_rois += stats.get('roi_count', 0)
                            total_pixels += stats.get('pixel_count', 0)
                            all_ff_values.append(stats.get('mean_fat_fraction', 0))
                
                f.write(f"总ROI数量: {total_rois}\n")
                f.write(f"总像素数量: {total_pixels:,}\n")
                
                if all_ff_values:
                    f.write(f"平均脂肪分数: {np.mean(all_ff_values):.3f}\n")
                    f.write(f"脂肪分数标准差: {np.std(all_ff_values):.3f}\n")
                    f.write(f"最小脂肪分数: {np.min(all_ff_values):.3f}\n")
                    f.write(f"最大脂肪分数: {np.max(all_ff_values):.3f}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
                
                # 按文件统计
                f.write("按文件统计:\n")
                for i, result in enumerate(self.batch_results, 1):
                    f.write(f"\n{i}. {result.get('image_file', 'unknown')}\n")
                    f.write(f"   总平均脂肪分数: {result.get('mean_fat_fraction', 0):.3f}\n")
                    f.write(f"   总像素数量: {result.get('pixel_count', 0):,}\n")
                    
                    if 'label_stats' in result:
                        f.write("   标签统计:\n")
                        for label_value, stats in result['label_stats'].items():
                            f.write(f"     标签 {label_value}: FF={stats.get('mean_fat_fraction', 0):.3f}, "
                                   f"像素={stats.get('pixel_count', 0):,}, ROI数={stats.get('roi_count', 0)}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                from datetime import datetime
                f.write("报告生成时间: " + str(datetime.now()) + "\n")
                
        except Exception as e:
            logger.error(f"创建统计报告失败: {str(e)}")
