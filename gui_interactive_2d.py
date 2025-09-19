"""
交互式2D模式GUI模块
处理交互式2D图像查看和ROI绘制
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from utils import logger, natural_sort_key
from gui_base import BaseGUI

class Interactive2DGUI(BaseGUI):
    """交互式2D模式GUI"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # 交互式模式相关
        self.interactive_image_files = []
        self.interactive_current_index = 0
        self.interactive_images_dir = None
        self.image_roi_dict = {}  # 存储每张图像的ROI数据
        
        # 状态变量
        self.status_var = tk.StringVar(value="交互式2D模式：请选择图像")
        
    def create_interactive_toolbar(self):
        """创建交互式模式工具栏"""
        # ROI类型选择
        ttk.Label(self.toolbar, text="ROI类型:").pack(side=tk.LEFT, padx=(0, 5))
        roi_type_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_type_var, 
                                     values=["rectangle", "circle", "polygon"], state="readonly", width=10)
        roi_type_combo.pack(side=tk.LEFT, padx=(0, 5))
        roi_type_combo.bind("<<ComboboxSelected>>", self.on_roi_type_change)
        
        # 图像操作按钮
        ttk.Button(self.toolbar, text="打开图像文件夹", command=self.open_interactive_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="打开图像", command=self.open_ff_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="打开ROI掩码", command=self.open_roi_mask).pack(side=tk.LEFT, padx=(0, 5))
        
        
        # 图像导航
        ttk.Button(self.toolbar, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=(0, 5))
        
        # ROI操作按钮
        ttk.Button(self.toolbar, text="删除ROI", command=self.delete_last_roi).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="清除所有", command=self.clear_all_roi).pack(side=tk.LEFT, padx=(0, 5))
        
        # 保存按钮（统一保存功能）
        ttk.Button(self.toolbar, text="保存", command=self.save_current_image_with_roi).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="设置输出目录", command=self.set_output_directory).pack(side=tk.LEFT, padx=(0, 5))
        
        
        # 图像信息显示
        self.image_index_var = tk.StringVar(value="0/0")
        ttk.Label(self.toolbar, textvariable=self.image_index_var).pack(side=tk.RIGHT, padx=(5, 0))
    
    def handle_mode_specific_keys(self, event):
        """处理交互式模式特定的快捷键"""
        if event.keysym in ["Left", "a", "A"]:
            self.prev_image()
        elif event.keysym in ["Right", "d", "D"]:
            self.next_image()
        elif event.state & 0x1 and event.keysym == "s":  # Ctrl+Shift+S
            self.save_current_image_roi()
    
    def open_interactive_folder(self):
        """打开交互式图像文件夹"""
        folder_path = filedialog.askdirectory(title="选择图像文件夹")
        if folder_path:
            self.interactive_images_dir = folder_path
            self.load_interactive_images()
    
    def load_interactive_images(self):
        """加载交互式图像列表"""
        if not self.interactive_images_dir:
            return
        
        # 获取所有图像文件
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        self.interactive_image_files = []
        
        for file in os.listdir(self.interactive_images_dir):
            if file.lower().endswith(image_extensions):
                self.interactive_image_files.append(file)
        
        # 使用自然排序
        self.interactive_image_files.sort(key=natural_sort_key)
        
        if self.interactive_image_files:
            self.interactive_current_index = 0
            self.load_interactive_image()
            self.status_var.set(f"已加载 {len(self.interactive_image_files)} 张图像")
        else:
            messagebox.showwarning("警告", "文件夹中没有找到图像文件")
    
    def load_interactive_image(self):
        """加载当前交互式图像"""
        if not self.interactive_image_files:
            return
        
        current_file = self.interactive_image_files[self.interactive_current_index]
        image_path = os.path.join(self.interactive_images_dir, current_file)
        
        try:
            # 加载图像
            self.ff_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if self.ff_image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 检查是否有对应的ROI数据
            if current_file in self.image_roi_dict:
                self.roi_list = self.image_roi_dict[current_file].copy()
                logger.info(f"加载图像 {current_file} 的ROI数据: {len(self.roi_list)} 个ROI")
            else:
                self.roi_list = []
                logger.info(f"图像 {current_file} 没有ROI数据")
            
            # 重置旋转角度
            self.rotation_angle = 0
            
            # 显示图像
            self.display_image()
            self.update_image_info()
            self.update_roi_info()
            self.update_image_index()
            
            self.status_var.set(f"已加载: {current_file}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
            logger.error(f"加载图像失败: {str(e)}")
    
    def prev_image(self):
        """上一张图像"""
        if not self.interactive_image_files:
            return
        
        # 保存当前图像的ROI数据
        if self.interactive_image_files and self.roi_list:
            current_file = self.interactive_image_files[self.interactive_current_index]
            self.image_roi_dict[current_file] = self.roi_list.copy()
            logger.info(f"保存图像 {current_file} 的ROI数据: {len(self.roi_list)} 个ROI")
        
        # 切换到上一张图像
        self.interactive_current_index = (self.interactive_current_index - 1) % len(self.interactive_image_files)
        self.load_interactive_image()
    
    def next_image(self):
        """下一张图像"""
        if not self.interactive_image_files:
            return
        
        # 保存当前图像的ROI数据
        if self.interactive_image_files and self.roi_list:
            current_file = self.interactive_image_files[self.interactive_current_index]
            self.image_roi_dict[current_file] = self.roi_list.copy()
            logger.info(f"保存图像 {current_file} 的ROI数据: {len(self.roi_list)} 个ROI")
        
        # 切换到下一张图像
        self.interactive_current_index = (self.interactive_current_index + 1) % len(self.interactive_image_files)
        self.load_interactive_image()
    
    def save_current_image_roi(self):
        """保存当前图像的ROI数据"""
        if not self.interactive_image_files or not self.roi_list:
            messagebox.showwarning("警告", "没有ROI数据可保存")
            return
        
        current_file = self.interactive_image_files[self.interactive_current_index]
        self.image_roi_dict[current_file] = self.roi_list.copy()
        
        # 保存到文件
        if self.output_directory:
            self.save_current_image_with_roi()
        
        self.status_var.set(f"已保存 {current_file} 的ROI数据")
        messagebox.showinfo("成功", f"已保存 {current_file} 的ROI数据")
    
    def open_ff_image(self):
        """打开FF图像"""
        file_path = filedialog.askopenfilename(
            title="选择FF图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.ff_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.ff_image is None:
                    raise ValueError(f"无法加载图像: {file_path}")
                
                # 清空ROI数据
                self.roi_list = []
                self.rotation_angle = 0
                
                # 显示图像
                self.display_image()
                self.update_image_info()
                self.update_roi_info()
                
                self.status_var.set(f"已加载: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {str(e)}")
                logger.error(f"加载图像失败: {str(e)}")
    
    def open_roi_mask(self):
        """打开ROI掩码"""
        file_path = filedialog.askopenfilename(
            title="选择ROI掩码",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.roi_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.roi_mask is None:
                    raise ValueError(f"无法加载ROI掩码: {file_path}")
                
                # 将掩码转换为ROI列表
                self.convert_mask_to_roi_list(self.roi_mask)
                
                # 显示图像
                self.display_image()
                self.update_roi_info()
                
                self.status_var.set(f"已加载ROI掩码: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载ROI掩码失败: {str(e)}")
                logger.error(f"加载ROI掩码失败: {str(e)}")
    
    def display_image(self):
        """显示图像"""
        if self.ff_image is None:
            return
        
        # 应用旋转
        if self.rotation_angle != 0:
            display_image = self.apply_rotation(self.ff_image, self.rotation_angle)
        else:
            display_image = self.ff_image.copy()
        
        # 转换为RGB
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        
        # 绘制ROI
        self.draw_rois_on_image(display_image)
        
        # 绘制临时多边形（如果正在绘制）
        if self.current_roi_type == 'polygon' and len(self.polygon_points) > 0:
            self.draw_temp_polygon(display_image)
        
        # 转换为PIL图像并显示
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
        
        # 清空画布并显示图像
        self.canvas.delete("all")
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, 
                               anchor=tk.NW, image=self.image_tk)
        
        # 绘制ROI和脂肪分数
        self.draw_rois_on_canvas()
    
    def save_current_image_with_roi(self):
        """保存当前图像和ROI"""
        if self.ff_image is None:
            messagebox.showwarning("警告", "没有图像可保存")
            return
        
        if not self.output_directory:
            messagebox.showwarning("警告", "请先设置输出目录")
            return
        
        try:
            # 创建输出目录
            os.makedirs(self.output_directory, exist_ok=True)
            
            # 生成文件名
            if self.interactive_image_files:
                base_name = os.path.splitext(self.interactive_image_files[self.interactive_current_index])[0]
            else:
                base_name = "image"
            
            # 保存带ROI的图像
            self.save_2d_image_with_roi(base_name)
            
            # 保存ROI数据
            self.save_2d_json_data(base_name)
            
            self.status_var.set(f"已保存图像和ROI数据到: {self.output_directory}")
            messagebox.showinfo("成功", "图像和ROI数据已保存")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            logger.error(f"保存失败: {str(e)}")
    
    def save_2d_image_with_roi(self, base_name):
        """保存2D图像和ROI"""
        # 创建带ROI的图像
        image_with_roi = self.ff_image.copy()
        if len(image_with_roi.shape) == 2:
            image_with_roi = cv2.cvtColor(image_with_roi, cv2.COLOR_GRAY2BGR)
        
        # 绘制ROI
        for i, roi in enumerate(self.roi_list):
            color = (0, 255, 0)  # 绿色
            thickness = 2
            
            if roi['type'] == 'rectangle':
                cv2.rectangle(image_with_roi, roi['start'], roi['end'], color, thickness)
                
                # 计算中心点
                center_x = (roi['start'][0] + roi['end'][0]) // 2
                center_y = (roi['start'][1] + roi['end'][1]) // 2
                
                # 添加ROI标签和脂肪分数
                if 'fat_fraction' in roi and roi['fat_fraction']:
                    ff_value = roi['fat_fraction']['mean_fat_fraction']
                    text = f'ROI{i+1}: {ff_value:.3f}'
                else:
                    text = f'ROI{i+1}'
                
                # 在ROI中心显示文本
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                cv2.putText(image_with_roi, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 黄色
                
            elif roi['type'] == 'circle':
                cv2.circle(image_with_roi, roi['center'], roi['radius'], color, thickness)
                
                # 添加ROI标签和脂肪分数
                if 'fat_fraction' in roi and roi['fat_fraction']:
                    ff_value = roi['fat_fraction']['mean_fat_fraction']
                    text = f'ROI{i+1}: {ff_value:.3f}'
                else:
                    text = f'ROI{i+1}'
                
                # 在圆心显示文本
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = roi['center'][0] - text_size[0] // 2
                text_y = roi['center'][1] + text_size[1] // 2
                cv2.putText(image_with_roi, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 黄色
                
            elif roi['type'] == 'polygon':
                # 绘制多边形
                points = np.array(roi['points'], dtype=np.int32)
                cv2.polylines(image_with_roi, [points], True, color, thickness)
                
                # 计算多边形中心点
                center_x = int(np.mean([p[0] for p in roi['points']]))
                center_y = int(np.mean([p[1] for p in roi['points']]))
                
                # 添加ROI标签和脂肪分数
                if 'fat_fraction' in roi and roi['fat_fraction']:
                    ff_value = roi['fat_fraction']['mean_fat_fraction']
                    text = f'ROI{i+1}: {ff_value:.3f}'
                else:
                    text = f'ROI{i+1}'
                
                # 在多边形中心显示文本
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                cv2.putText(image_with_roi, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 黄色
        
        # 保存图像
        output_path = os.path.join(self.output_directory, f"{base_name}_with_roi.png")
        cv2.imwrite(output_path, image_with_roi)
        logger.info(f"已保存带ROI的图像: {output_path}")
    
    def save_2d_json_data(self, base_name):
        """保存2D JSON数据"""
        # 准备ROI数据
        roi_data = []
        for i, roi in enumerate(self.roi_list):
            roi_info = {
                'id': i + 1,
                'type': roi['type'],
                'fat_fraction': roi.get('fat_fraction'),
                'statistics': roi.get('statistics', {})
            }
            
            if roi['type'] == 'rectangle':
                roi_info['start'] = roi['start']
                roi_info['end'] = roi['end']
            elif roi['type'] == 'circle':
                roi_info['center'] = roi['center']
                roi_info['radius'] = roi['radius']
            elif roi['type'] == 'polygon':
                roi_info['points'] = roi['points']
                roi_info['vertex_count'] = len(roi['points'])
            
            roi_data.append(roi_info)
        
        # 保存JSON数据
        import json
        json_path = os.path.join(self.output_directory, f"{base_name}_roi_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image_file': self.interactive_image_files[self.interactive_current_index] if self.interactive_image_files else "unknown",
                'roi_count': len(roi_data),
                'roi_data': roi_data,
                'rotation_angle': self.rotation_angle
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已保存ROI数据: {json_path}")
