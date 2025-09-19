"""
批量处理2D模式GUI模块
处理批量2D图像处理和分析
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from utils import logger, natural_sort_key
from gui_base import BaseGUI
from analyzer import batch_analyze_ff_images

class Batch2DGUI(BaseGUI):
    """批量处理2D模式GUI"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # 批量处理相关
        self.batch_images_dir = None
        self.batch_labels_dir = None
        self.batch_image_files = []
        
        # 状态变量
        self.status_var = tk.StringVar(value="批量处理2D模式：请选择图像文件夹")
    
    def display_image(self):
        """显示批量处理图像（基础版本）"""
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
        self.draw_rois_on_image()
        self.batch_current_index = 0
        self.batch_results = []
        self.batch_processing = False
        
    def create_batch_toolbar(self):
        """创建批量处理模式工具栏"""
        # 文件夹选择
        ttk.Button(self.toolbar, text="选择图像文件夹", command=self.open_batch_images_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="选择标签文件夹", command=self.open_batch_labels_folder).pack(side=tk.LEFT, padx=(0, 5))
        
        
        # 图像导航
        ttk.Button(self.toolbar, text="上一张", command=self.prev_batch_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="下一张", command=self.next_batch_image).pack(side=tk.LEFT, padx=(0, 5))
        
        # 处理控制
        self.process_button = ttk.Button(self.toolbar, text="开始处理", command=self.start_batch_processing)
        self.process_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(self.toolbar, text="停止处理", command=self.stop_batch_processing).pack(side=tk.LEFT, padx=(0, 5))
        
        # 结果操作
        ttk.Button(self.toolbar, text="导出结果", command=self.export_batch_results).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="设置输出目录", command=self.set_output_directory).pack(side=tk.LEFT, padx=(0, 5))
        
        
        # 图像信息显示
        self.batch_index_var = tk.StringVar(value="0/0")
        ttk.Label(self.toolbar, textvariable=self.batch_index_var).pack(side=tk.RIGHT, padx=(5, 0))
    
    def handle_mode_specific_keys(self, event):
        """处理批量处理模式特定的快捷键"""
        if event.keysym in ["Left", "a", "A"]:
            self.prev_batch_image()
        elif event.keysym in ["Right", "d", "D"]:
            self.next_batch_image()
        elif event.keysym == "space":
            if self.batch_processing:
                self.stop_batch_processing()
            else:
                self.start_batch_processing()
        elif event.state & 0x4 and event.keysym == "e":  # Ctrl+E
            self.export_batch_results()
    
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
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        self.batch_image_files = []
        
        for file in os.listdir(self.batch_images_dir):
            if file.lower().endswith(image_extensions):
                self.batch_image_files.append(file)
        
        # 使用自然排序
        self.batch_image_files.sort(key=natural_sort_key)
        
        if self.batch_image_files:
            self.batch_current_index = 0
            self.show_first_batch_image()
            self.status_var.set(f"已加载 {len(self.batch_image_files)} 张图像")
        else:
            messagebox.showwarning("警告", "文件夹中没有找到图像文件")
    
    def show_first_batch_image(self):
        """显示第一张批量图像"""
        if self.batch_image_files:
            self.batch_current_index = 0
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
            # 使用analyzer模块的批量处理功能
            self.batch_results = batch_analyze_ff_images(
                self.batch_images_dir, 
                self.batch_labels_dir, 
                self.output_directory
            )
            
            # 更新UI
            self.root.after(0, self.batch_processing_completed)
            
        except Exception as e:
            logger.error(f"批量处理失败: {str(e)}")
            self.root.after(0, lambda: self.batch_processing_error(str(e)))
    
    def batch_processing_completed(self):
        """批量处理完成"""
        self.batch_processing = False
        self.process_button.config(text="开始处理")
        self.status_var.set(f"批量处理完成，共处理 {len(self.batch_results)} 张图像")
        messagebox.showinfo("完成", f"批量处理完成，共处理 {len(self.batch_results)} 张图像")
    
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
        
        self.batch_current_index = (self.batch_current_index - 1) % len(self.batch_image_files)
        self.show_batch_image_by_index(self.batch_current_index)
    
    def next_batch_image(self):
        """下一张批量图像"""
        if not self.batch_image_files:
            return
        
        self.batch_current_index = (self.batch_current_index + 1) % len(self.batch_image_files)
        self.show_batch_image_by_index(self.batch_current_index)
    
    def show_batch_image_by_index(self, index):
        """显示指定索引的批量图像"""
        if not self.batch_image_files or index >= len(self.batch_image_files):
            return
        
        try:
            # 加载图像
            image_file = self.batch_image_files[index]
            image_path = os.path.join(self.batch_images_dir, image_file)
            
            self.ff_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if self.ff_image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 查找对应的标签文件
            label_path = self.find_corresponding_label(image_file)
            if label_path:
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
            
            # 显示图像
            self.display_image()
            self.update_image_info()
            self.update_roi_info()
            self.update_batch_index()
            
            # 显示处理结果（如果有）
            if index < len(self.batch_results):
                self.display_batch_result(index)
            
            self.status_var.set(f"显示图像: {image_file}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
            logger.error(f"加载图像失败: {str(e)}")
    
    def find_corresponding_label(self, image_file):
        """查找对应的标签文件"""
        if not self.batch_labels_dir:
            return None
        
        # 尝试直接匹配
        base_name = os.path.splitext(image_file)[0]
        label_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        
        for ext in label_extensions:
            label_file = f"{base_name}_roi{ext}"
            label_path = os.path.join(self.batch_labels_dir, label_file)
            if os.path.exists(label_path):
                return label_path
        
        # 尝试模糊匹配
        return self.fuzzy_match_label(image_file, label_extensions)
    
    def fuzzy_match_label(self, image_file, label_extensions):
        """模糊匹配标签文件"""
        if not self.batch_labels_dir:
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
    
    def convert_mask_to_roi_list(self, roi_mask):
        """将掩码转换为ROI列表"""
        # 查找轮廓
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.roi_list = []
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
            self.roi_list.append(roi)
        
        logger.info(f"从掩码转换得到 {len(self.roi_list)} 个ROI")
    
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
            self.batch_index_var.set(f"{self.batch_current_index + 1}/{len(self.batch_image_files)}")
        else:
            self.batch_index_var.set("0/0")
    
    def set_output_directory(self):
        """设置输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_directory = directory
            self.status_var.set(f"输出目录已设置为: {directory}")
    
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
                    '图像文件': result.get('image_file', 'unknown'),
                    '平均脂肪分数': result.get('mean_fat_fraction', 0),
                    '标准差': result.get('std_fat_fraction', 0),
                    '中位数': result.get('median_fat_fraction', 0),
                    '像素数量': result.get('pixel_count', 0),
                    'ROI面积': result.get('roi_area', 0),
                    '覆盖率(%)': result.get('roi_coverage_percent', 0)
                })
            
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.output_directory, "batch_results_summary.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # 保存JSON结果
            import json
            json_path = os.path.join(self.output_directory, "batch_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.batch_results, f, indent=2, ensure_ascii=False)
            
            self.status_var.set(f"结果已导出到: {self.output_directory}")
            messagebox.showinfo("成功", f"结果已导出到: {self.output_directory}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")
            logger.error(f"导出失败: {str(e)}")
