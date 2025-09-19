"""
交互式3D模式GUI模块
处理交互式3D NIfTI文件查看和ROI绘制
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from utils import logger, natural_sort_key, NIBABEL_AVAILABLE
from gui_base import BaseGUI
from nifti_viewer import NIfTI3DViewer, load_nifti_image

class Interactive3DGUI(BaseGUI):
    """交互式3D模式GUI"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # 3D视图控制变量
        self.current_3d_viewer = None  # 当前3D查看器实例
        self.current_view = 'axial'  # 当前视图：axial, sagittal, coronal
        self.current_slice = 0  # 当前切片索引
        self.is_3d_mode = False  # 是否为3D模式
        self.image_roi_dict = {}  # 存储每张图像的ROI数据
        self.slice_roi_dict = {}  # 存储每个切片的ROI数据 {view_slice_key: roi_list}
        
        # 交互式模式相关（用于兼容性）
        self.interactive_image_files = []
        self.interactive_current_index = 0
        self.interactive_images_dir = None
        
        # 状态变量
        self.status_var = tk.StringVar(value="交互式3D模式：请选择3D文件")
        
        # ROI绘制相关变量（继承基类但需要确保初始化）
        self.current_roi_type = 'rectangle'
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.center_point = None
        self.radius = 0
        self.polygon_points = []
        self.polygon_drawing = False
        
        # 图像显示相关变量（确保初始化）
        self.scale_factor = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
    
    def get_slice_key(self, view=None, slice_idx=None):
        """获取切片键"""
        if view is None:
            view = self.current_view
        if slice_idx is None:
            slice_idx = self.current_slice
        return f"{view}_{slice_idx}"
    
    def reverse_rotate_coordinates(self, points, rotation_angle):
        """反向旋转坐标（从显示坐标转换回原始坐标）"""
        if rotation_angle == 0:
            return points
        
        # 获取原始图像尺寸
        if self.ff_image is not None:
            h, w = self.ff_image.shape[:2]
        else:
            return points
        
        reversed_points = []
        for point in points:
            x, y = point
            
            # 反向旋转坐标
            if rotation_angle == 90:
                # 90度反向：(x,y) -> (y, h-1-x)
                new_x = y
                new_y = h - 1 - x
            elif rotation_angle == 180:
                # 180度反向：(x,y) -> (w-1-x, h-1-y)
                new_x = w - 1 - x
                new_y = h - 1 - y
            elif rotation_angle == 270:
                # 270度反向：(x,y) -> (w-1-y, x)
                new_x = w - 1 - y
                new_y = x
            else:
                new_x, new_y = x, y
            
            reversed_points.append((new_x, new_y))
        
        return reversed_points
    
    def on_mouse_click(self, event):
        """鼠标点击事件（3D模式 - 支持旋转）"""
        if self.ff_image is None:
            return
            
        if self.current_roi_type == 'polygon':
            # 多边形模式：添加顶点
            # 考虑图像偏移量
            adjusted_x = event.x - self.image_offset_x
            adjusted_y = event.y - self.image_offset_y
            point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
            
            # 如果有旋转，需要反向转换坐标
            if self.rotation_angle != 0:
                point = self.reverse_rotate_coordinates([point], self.rotation_angle)[0]
            
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
            point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
            
            # 如果有旋转，需要反向转换坐标
            if self.rotation_angle != 0:
                point = self.reverse_rotate_coordinates([point], self.rotation_angle)[0]
            
            self.start_point = point
            self.end_point = point
            self.center_point = point
            self.radius = 0
    
    def on_mouse_drag(self, event):
        """鼠标拖拽事件（3D模式 - 支持旋转）"""
        if not self.drawing or self.ff_image is None:
            return
        
        # 考虑图像偏移量
        adjusted_x = event.x - self.image_offset_x
        adjusted_y = event.y - self.image_offset_y
        point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
        
        # 如果有旋转，需要反向转换坐标
        if self.rotation_angle != 0:
            point = self.reverse_rotate_coordinates([point], self.rotation_angle)[0]
        
        if self.current_roi_type == 'rectangle':
            self.end_point = point
        elif self.current_roi_type == 'circle':
            # 计算半径
            dx = point[0] - self.center_point[0]
            dy = point[1] - self.center_point[1]
            self.radius = int(np.sqrt(dx*dx + dy*dy))
        
        # 更新显示
        self.display_image()
    
    def on_mouse_release(self, event):
        """鼠标释放事件（3D模式 - 支持旋转）"""
        if not self.drawing or self.ff_image is None:
            return
        
        # 考虑图像偏移量
        adjusted_x = event.x - self.image_offset_x
        adjusted_y = event.y - self.image_offset_y
        point = (int(adjusted_x / self.scale_factor), int(adjusted_y / self.scale_factor))
        
        # 如果有旋转，需要反向转换坐标
        if self.rotation_angle != 0:
            point = self.reverse_rotate_coordinates([point], self.rotation_angle)[0]
        
        if self.current_roi_type == 'rectangle':
            self.end_point = point
        elif self.current_roi_type == 'circle':
            # 计算半径
            dx = point[0] - self.center_point[0]
            dy = point[1] - self.center_point[1]
            self.radius = int(np.sqrt(dx*dx + dy*dy))
        
        # 保持绘制状态，显示红色ROI，等待S键保存
        # self.drawing 保持 True，直到用户按S键保存
        
        # 更新显示，显示红色ROI
        self.display_image()
    
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
    
    def save_current_roi(self):
        """保存当前ROI（3D模式）"""
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
            messagebox.showwarning("警告", "请先绘制ROI区域")
            return
        
        # 添加到当前切片的ROI列表
        slice_key = self.get_slice_key()
        if slice_key not in self.slice_roi_dict:
            self.slice_roi_dict[slice_key] = []
        self.slice_roi_dict[slice_key].append(roi)
        
        # 同时更新全局ROI列表（用于显示）
        self.roi_list.append(roi)
        
        # 计算脂肪分数
        self.calculate_roi_fat_fraction(roi)
        
        # 清空绘制状态
        self.start_point = None
        self.end_point = None
        self.center_point = None
        self.radius = 0
        self.drawing = False
        
        # 更新显示
        self.display_image()
        self.update_output_display()
    
    def save_polygon_roi(self):
        """保存多边形ROI（3D模式）"""
        if len(self.polygon_points) < 3:
            messagebox.showwarning("警告", "多边形至少需要3个顶点")
            return
        
        # 创建多边形ROI
        roi = {
            'type': 'polygon',
            'points': self.polygon_points.copy()
        }
        
        # 添加到当前切片的ROI列表
        slice_key = self.get_slice_key()
        if slice_key not in self.slice_roi_dict:
            self.slice_roi_dict[slice_key] = []
        self.slice_roi_dict[slice_key].append(roi)
        
        # 同时更新全局ROI列表（用于显示）
        self.roi_list.append(roi)
        
        # 计算脂肪分数
        self.calculate_roi_fat_fraction(roi)
        
        # 清空多边形绘制状态
        self.polygon_points = []
        self.polygon_drawing = False
        
        # 更新显示
        self.display_image()
        self.update_output_display()
    
    def draw_current_roi(self):
        """绘制当前正在绘制的ROI（3D模式 - 支持旋转）"""
        if not self.drawing or self.ff_image is None:
            return
        
        # 获取原始图像
        temp_image = self.ff_image.copy()
        if len(temp_image.shape) == 2:
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
        
        # 应用旋转（如果有）
        if self.rotation_angle != 0:
            temp_image = self.apply_rotation(temp_image, self.rotation_angle)
        
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
        
        # 绘制临时多边形
        if self.current_roi_type == 'polygon' and len(self.polygon_points) > 0:
            self.draw_temp_polygon(temp_image)
        
        # 转换为PIL图像并显示
        if len(temp_image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(temp_image)
        
        # 调整图像大小
        if hasattr(self, 'scale_factor') and self.scale_factor != 1.0:
            new_width = int(temp_image.shape[1] * self.scale_factor)
            new_height = int(temp_image.shape[0] * self.scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter图像
        self.image_tk = ImageTk.PhotoImage(pil_image)
        
        # 在画布上显示图像
        self.canvas.delete("all")
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, 
                               anchor=tk.NW, image=self.image_tk)
    
    def draw_rois_on_image(self, image):
        """在图像上绘制ROI（3D模式 - 支持旋转）"""
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
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # 计算中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
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
                cv2.putText(image, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 黄色
                
            elif roi['type'] == 'circle':
                cx = int(center[0])
                cy = int(center[1])
                r = int(radius)
                cv2.circle(image, (cx, cy), r, color, thickness)
                
                # 添加ROI标签和脂肪分数
                if 'fat_fraction' in roi and roi['fat_fraction']:
                    ff_value = roi['fat_fraction']['mean_fat_fraction']
                    text = f'ROI{i+1}: {ff_value:.3f}'
                else:
                    text = f'ROI{i+1}'
                
                # 在ROI中心显示文本
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                cv2.putText(image, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 黄色
                
            elif roi['type'] == 'polygon':
                # 绘制多边形
                points = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
                cv2.polylines(image, [points], True, color, thickness)
                
                # 计算中心点
                center_x = int(np.mean([p[0] for p in points]))
                center_y = int(np.mean([p[1] for p in points]))
                
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
                cv2.putText(image, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 黄色

    def draw_temp_polygon(self, image):
        """绘制临时多边形（正在绘制的多边形 - 支持旋转）"""
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
    
    def display_image(self):
        """显示3D图像切片"""
        if self.current_3d_viewer is None:
            return
        
        try:
            # 获取当前切片
            slice_data = self.current_3d_viewer.get_slice(self.current_view, self.current_slice)
            if slice_data is None:
                return
            
            # 设置ff_image（基类需要）
            self.ff_image = slice_data
            
            # 转换为RGB图像用于显示
            if len(slice_data.shape) == 2:
                display_image = cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR)
            else:
                display_image = slice_data.copy()
            
            # 应用旋转
            if self.rotation_angle != 0:
                display_image = self.apply_rotation(display_image, self.rotation_angle)
            
            # 绘制ROI到图像上
            self.draw_rois_on_image(display_image)
            
            # 绘制当前正在绘制的ROI（红色提示框）
            if self.drawing:
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
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色
                elif self.current_roi_type == 'circle' and self.center_point and self.radius > 0:
                    # 根据旋转角度调整坐标
                    if self.rotation_angle != 0:
                        center = self.rotate_roi_coordinates_for_display([self.center_point], self.rotation_angle)[0]
                    else:
                        center = self.center_point
                    
                    cx = int(center[0])
                    cy = int(center[1])
                    r = int(self.radius)
                    cv2.circle(display_image, (cx, cy), r, (0, 0, 255), 2)  # 红色
            
            # 绘制临时多边形（如果正在绘制）
            if self.current_roi_type == 'polygon' and len(self.polygon_points) > 0:
                self.draw_temp_polygon(display_image)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            
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
            
            # 调试信息
            logger.debug(f"3D模式坐标信息: scale_factor={self.scale_factor}, offset_x={self.image_offset_x}, offset_y={self.image_offset_y}")
            logger.debug(f"3D模式图像尺寸: {display_image.shape}, 画布尺寸: {canvas_width}x{canvas_height}")
            
        except Exception as e:
            logger.error(f"显示3D切片失败: {str(e)}")
            if hasattr(self, 'status_var') and self.status_var:
                self.status_var.set(f"显示切片失败: {str(e)}")
        self.interactive_images_dir = None
        
    def create_interactive_toolbar(self):
        """创建交互式3D模式工具栏"""
        # 3D视图控制
        ttk.Label(self.toolbar, text="视图:").pack(side=tk.LEFT, padx=(0, 5))
        self.view_var = tk.StringVar(value="axial")
        view_combo = ttk.Combobox(self.toolbar, textvariable=self.view_var, 
                                 values=["axial", "sagittal", "coronal"], state="readonly", width=10)
        view_combo.pack(side=tk.LEFT, padx=(0, 5))
        view_combo.bind("<<ComboboxSelected>>", self.on_view_change)
        
        # 切片控制
        ttk.Label(self.toolbar, text="切片:").pack(side=tk.LEFT, padx=(0, 5))
        self.slice_var = tk.StringVar(value="0/0")
        ttk.Label(self.toolbar, textvariable=self.slice_var).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(self.toolbar, text="上一片", command=self.prev_slice).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="下一片", command=self.next_slice).pack(side=tk.LEFT, padx=(0, 5))
        
        # ROI类型选择
        ttk.Label(self.toolbar, text="ROI类型:").pack(side=tk.LEFT, padx=(0, 5))
        self.roi_type_var = tk.StringVar(value="rectangle")
        roi_type_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_type_var, 
                                     values=["rectangle", "circle", "polygon"], state="readonly", width=10)
        roi_type_combo.pack(side=tk.LEFT, padx=(0, 5))
        roi_type_combo.bind("<<ComboboxSelected>>", self.on_roi_type_change)
        
        # 图像操作按钮
        ttk.Button(self.toolbar, text="打开3D文件", command=self.open_3d_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="打开图像文件夹", command=self.open_interactive_folder).pack(side=tk.LEFT, padx=(0, 5))
        
        
        # ROI操作按钮
        ttk.Button(self.toolbar, text="删除ROI", command=self.delete_last_roi).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="清除所有", command=self.clear_all_roi).pack(side=tk.LEFT, padx=(0, 5))
        
        # 保存按钮（统一保存功能）
        ttk.Button(self.toolbar, text="保存", command=self.save_3d_roi_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.toolbar, text="设置输出目录", command=self.set_output_directory).pack(side=tk.LEFT, padx=(0, 5))
        
        
        # 图像信息显示
        self.image_index_var = tk.StringVar(value="0/0")
        ttk.Label(self.toolbar, textvariable=self.image_index_var).pack(side=tk.RIGHT, padx=(5, 0))
    
    def handle_mode_specific_keys(self, event):
        """处理交互式3D模式特定的快捷键"""
        if event.keysym in ["Left", "a", "A"]:
            self.prev_slice()
        elif event.keysym in ["Right", "d", "D"]:
            self.next_slice()
        elif event.keysym in ["Up", "w", "W"]:
            self.prev_slice()
        elif event.keysym in ["Down", "x", "X"]:
            self.next_slice()
        elif event.keysym in ["s", "S"]:
            # S键保存当前ROI
            self.save_current_roi()
        elif event.keysym == "v":
            # 切换视图
            views = ["axial", "sagittal", "coronal"]
            current_index = views.index(self.current_view)
            next_index = (current_index + 1) % len(views)
            self.view_var.set(views[next_index])
            self.on_view_change()
    
    def open_3d_file(self):
        """打开3D文件"""
        if not NIBABEL_AVAILABLE:
            messagebox.showerror("错误", "nibabel库未安装，无法处理3D文件")
            return
        
        file_path = filedialog.askopenfilename(
            title="选择3D文件",
            filetypes=[("NIfTI文件", "*.nii *.nii.gz"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 加载3D文件
                self.current_3d_viewer = NIfTI3DViewer(file_path)
                self.current_3d_viewer.load_header()
                
                # 设置默认视图和切片
                self.current_view = 'axial'
                self.current_slice = self.current_3d_viewer.current_slice
                
                # 设置为3D模式
                self.is_3d_mode = True
                
                # 加载当前切片
                self.load_current_3d_slice()
                
                self.status_var.set(f"已加载3D文件: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载3D文件失败: {str(e)}")
                logger.error(f"加载3D文件失败: {str(e)}")
    
    def on_view_change(self, event=None):
        """视图改变事件"""
        if not self.is_3d_mode or not self.current_3d_viewer:
            return
        
        self.current_view = self.view_var.get()
        self.current_3d_viewer.set_view(self.current_view)
        
        # 重置切片索引
        view_info = self.current_3d_viewer.get_view_info(self.current_view)
        if view_info:
            self.current_slice = view_info['max_slices'] // 2
            self.current_3d_viewer.set_slice(self.current_slice)
        
        # 加载新视图的切片
        self.load_current_3d_slice()
        
        self.status_var.set(f"切换到{self.current_view}视图")
    
    def on_slice_change(self, event=None):
        """切片改变事件"""
        if not self.is_3d_mode or not self.current_3d_viewer:
            return
        
        try:
            slice_index = int(self.slice_var.get())
            self.current_slice = slice_index
            self.current_3d_viewer.set_slice(slice_index)
            self.load_current_3d_slice()
        except ValueError:
            self.status_var.set("无效的切片索引")
    
    def prev_slice(self):
        """上一片"""
        if not self.is_3d_mode or not self.current_3d_viewer:
            return
        
        view_info = self.current_3d_viewer.get_view_info(self.current_view)
        if view_info and self.current_slice > 0:
            self.current_slice -= 1
            self.current_3d_viewer.set_slice(self.current_slice)
            self.load_current_3d_slice()
    
    def next_slice(self):
        """下一片"""
        if not self.is_3d_mode or not self.current_3d_viewer:
            return
        
        view_info = self.current_3d_viewer.get_view_info(self.current_view)
        if view_info and self.current_slice < view_info['max_slices'] - 1:
            self.current_slice += 1
            self.current_3d_viewer.set_slice(self.current_slice)
            self.load_current_3d_slice()
    
    def load_current_3d_slice(self):
        """加载当前3D切片"""
        if not self.is_3d_mode or not self.current_3d_viewer:
            return
        
        try:
            # 获取当前切片
            slice_data = self.current_3d_viewer.get_slice(self.current_view, self.current_slice)
            
            # 转换为8位图像
            if slice_data.dtype != np.uint8:
                slice_data = ((slice_data - slice_data.min()) / 
                            (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            self.ff_image = slice_data
            
            # 更新切片信息显示
            view_info = self.current_3d_viewer.get_view_info(self.current_view)
            if view_info:
                self.slice_var.set(f"{self.current_slice + 1}/{view_info['max_slices']}")
            
            # 加载当前切片的ROI数据
            slice_key = self.get_slice_key()
            if slice_key in self.slice_roi_dict:
                self.roi_list = self.slice_roi_dict[slice_key].copy()
                logger.info(f"加载切片 {slice_key} 的ROI数据: {len(self.roi_list)} 个ROI")
            else:
                self.roi_list = []
                logger.info(f"切片 {slice_key} 没有ROI数据")
            
            # 显示图像
            self.display_image()
            self.update_image_info()
            self.update_roi_info()
            
        except Exception as e:
            logger.error(f"加载3D切片失败: {str(e)}")
            self.status_var.set(f"加载切片失败: {str(e)}")
    
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
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.nii', '.nii.gz')
        self.interactive_image_files = []
        
        for file in os.listdir(self.interactive_images_dir):
            if file.lower().endswith(image_extensions):
                self.interactive_image_files.append(file)
        
        # 使用自然排序
        self.interactive_image_files.sort(key=natural_sort_key)
        
        if self.interactive_image_files:
            self.interactive_current_index = 0
            self.load_interactive_image()
            self.status_var.set(f"已加载 {len(self.interactive_image_files)} 个文件")
        else:
            messagebox.showwarning("警告", "文件夹中没有找到图像文件")
    
    def load_interactive_image(self):
        """加载当前交互式图像"""
        if not self.interactive_image_files:
            return
        
        current_file = self.interactive_image_files[self.interactive_current_index]
        file_path = os.path.join(self.interactive_images_dir, current_file)
        
        try:
            # 检查是否为3D文件
            if current_file.lower().endswith(('.nii', '.nii.gz')):
                if not NIBABEL_AVAILABLE:
                    messagebox.showerror("错误", "nibabel库未安装，无法处理3D文件")
                    return
                
                # 加载3D文件
                self.current_3d_viewer = NIfTI3DViewer(file_path)
                self.current_3d_viewer.load_header()
                
                # 设置默认视图和切片
                self.current_view = 'axial'
                self.current_slice = self.current_3d_viewer.current_slice
                
                # 加载当前切片
                self.load_current_3d_slice()
                
                self.is_3d_mode = True
                
            else:
                # 加载2D图像
                self.ff_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.ff_image is None:
                    raise ValueError(f"无法加载图像: {file_path}")
                
                self.is_3d_mode = False
                self.current_3d_viewer = None
                
                # 显示图像
                self.display_image()
                self.update_image_info()
                self.update_roi_info()
            
            # 检查是否有对应的ROI数据
            if current_file in self.image_roi_dict:
                self.roi_list = self.image_roi_dict[current_file].copy()
                logger.info(f"加载文件 {current_file} 的ROI数据: {len(self.roi_list)} 个ROI")
            else:
                self.roi_list = []
                logger.info(f"文件 {current_file} 没有ROI数据")
            
            # 重置旋转角度
            self.rotation_angle = 0
            
            # 更新显示
            self.update_image_index()
            
            self.status_var.set(f"已加载: {current_file}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")
            logger.error(f"加载文件失败: {str(e)}")
    
    def prev_image(self):
        """上一张图像"""
        if not self.interactive_image_files:
            return
        
        # 保存当前文件的ROI数据
        if self.interactive_image_files and self.roi_list:
            current_file = self.interactive_image_files[self.interactive_current_index]
            self.image_roi_dict[current_file] = self.roi_list.copy()
            logger.info(f"保存文件 {current_file} 的ROI数据: {len(self.roi_list)} 个ROI")
        
        # 切换到上一张图像
        self.interactive_current_index = (self.interactive_current_index - 1) % len(self.interactive_image_files)
        self.load_interactive_image()
    
    def next_image(self):
        """下一张图像"""
        if not self.interactive_image_files:
            return
        
        # 保存当前文件的ROI数据
        if self.interactive_image_files and self.roi_list:
            current_file = self.interactive_image_files[self.interactive_current_index]
            self.image_roi_dict[current_file] = self.roi_list.copy()
            logger.info(f"保存文件 {current_file} 的ROI数据: {len(self.roi_list)} 个ROI")
        
        # 切换到下一张图像
        self.interactive_current_index = (self.interactive_current_index + 1) % len(self.interactive_image_files)
        self.load_interactive_image()
    
    def save_current_image_with_roi(self):
        """保存3D ROI数据（智能批量保存）"""
        if not self.output_directory:
            messagebox.showwarning("警告", "请先设置输出目录")
            return
        
        # 检查是否有任何ROI数据
        has_roi_data = False
        for slice_key, roi_list in self.slice_roi_dict.items():
            if roi_list:
                has_roi_data = True
                break
        
        if not has_roi_data:
            messagebox.showwarning("警告", "没有ROI数据可保存")
            return
        
        try:
            # 创建输出目录
            os.makedirs(self.output_directory, exist_ok=True)
            
            # 生成基础文件名
            if self.interactive_image_files:
                base_name = os.path.splitext(self.interactive_image_files[self.interactive_current_index])[0]
            else:
                base_name = f"3d_data_{self.current_view}"
            
            saved_count = 0
            
            # 批量保存所有已标注的切片
            for slice_key, roi_list in self.slice_roi_dict.items():
                if not roi_list:  # 跳过没有ROI的切片
                    continue
                
                # 解析切片键
                view, slice_idx = slice_key.split('_', 1)
                slice_idx = int(slice_idx)
                
                # 临时设置当前切片和ROI列表用于保存
                original_slice = self.current_slice
                original_view = self.current_view
                original_roi_list = self.roi_list.copy()
                
                self.current_slice = slice_idx
                self.current_view = view
                self.roi_list = roi_list.copy()
                
                # 生成切片文件名
                slice_base_name = f"{base_name}_{view}_{slice_idx:03d}"
                
                # 保存切片图像
                self.save_3d_slice_image(slice_base_name)
                
                # 保存切片JSON数据
                self.save_3d_json_data(slice_base_name)
                
                saved_count += 1
                
                # 恢复原始状态
                self.current_slice = original_slice
                self.current_view = original_view
                self.roi_list = original_roi_list
            
            self.status_var.set(f"已批量保存 {saved_count} 个切片的ROI数据到: {self.output_directory}")
            messagebox.showinfo("成功", f"已批量保存 {saved_count} 个切片的ROI数据")
            logger.info(f"批量保存完成，共保存 {saved_count} 个切片")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            logger.error(f"保存失败: {str(e)}")
    
    def save_3d_roi_data(self):
        """保存3D ROI数据（兼容性方法）"""
        self.save_current_image_with_roi()
    
    def save_3d_slice_image(self, base_name):
        """保存3D切片图像"""
        if self.ff_image is None:
            return
        
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
        logger.info(f"已保存带ROI的3D切片图像: {output_path}")
    
    def save_3d_json_data(self, base_name):
        """保存3D JSON数据"""
        # 准备ROI数据
        roi_data = []
        for i, roi in enumerate(self.roi_list):
            roi_info = {
                'id': i + 1,
                'type': roi['type'],
                'fat_fraction': roi.get('fat_fraction'),
                'statistics': roi.get('statistics', {}),
                'view': self.current_view,
                'slice': self.current_slice
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
        json_path = os.path.join(self.output_directory, f"{base_name}_3d_roi_data.json")
        
        # 确保所有数据都是JSON可序列化的
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
        json_data = {
            'file_name': self.interactive_image_files[self.interactive_current_index] if self.interactive_image_files else "unknown",
            'is_3d_mode': self.is_3d_mode,
            'current_view': self.current_view,
            'current_slice': self.current_slice,
            'roi_count': len(roi_data),
            'roi_data': convert_numpy_types(roi_data),
            'rotation_angle': self.rotation_angle
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已保存3D ROI数据: {json_path}")
    
    def rotate_image_90(self):
        """旋转图像90度"""
        if self.ff_image is not None:
            self.rotation_angle = (self.rotation_angle + 90) % 360
            self.display_image()
            self.status_var.set(f"图像已旋转 {self.rotation_angle}°")
