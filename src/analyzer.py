"""
核心分析器模块
包含FF图像脂肪分数分析的核心功能
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from utils import logger, natural_sort_key

class FFImageAnalyzer:
    """FF图像脂肪分数分析器"""
    
    def __init__(self, ff_image_path, roi_mask_path=None):
        """
        初始化FF图像分析器
        
        Args:
            ff_image_path: FF图像文件路径
            roi_mask_path: ROI掩码文件路径（可选，如果不提供则需要手动勾画）
        """
        self.ff_image_path = ff_image_path
        self.roi_mask_path = roi_mask_path
        self.ff_image = None
        self.roi_mask = None
        self.fat_fraction_values = None
        self.results = {}
        
    def load_images(self):
        """加载FF图像和ROI掩码"""
        print("正在加载图像...")
        
        # 加载FF图像
        self.ff_image = cv2.imread(self.ff_image_path, cv2.IMREAD_GRAYSCALE)
        if self.ff_image is None:
            raise ValueError(f'无法加载FF图像: {self.ff_image_path}')
        
        # 加载ROI掩码（如果提供）
        if self.roi_mask_path and os.path.exists(self.roi_mask_path):
            self.roi_mask = cv2.imread(self.roi_mask_path, cv2.IMREAD_GRAYSCALE)
            if self.roi_mask is None:
                print(f'无法加载ROI掩码: {self.roi_mask_path}')
                self.roi_mask = None
        else:
            print("未提供ROI掩码，将使用手动勾画功能")
            self.roi_mask = None
            
        print(f'FF图像加载成功: {self.ff_image.shape}')
        if self.roi_mask is not None:
            print(f'ROI掩码加载成功: {self.roi_mask.shape}')
    
    def create_manual_roi(self):
        """手动创建ROI掩码（交互式图形界面）"""
        print("开始手动勾画ROI区域...")
        print("操作说明：")
        print("1. 按 'r' 切换到矩形模式，按 'c' 切换到圆形模式，按 'p' 切换到多边形模式")
        print("2. 矩形/圆形：在图像上拖拽鼠标绘制ROI")
        print("3. 多边形：点击绘制顶点，首尾两点自动闭合")
        print("4. 按 's' 保存当前ROI，按 'd' 删除最后一个ROI")
        print("5. 按 'q' 实时计算当前ROI结果")
        print("6. 按 'Enter' 确认选择并计算，按 'Esc' 取消选择")
        
        # 创建窗口
        cv2.namedWindow('FF Image - ROI绘制工具', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FF Image - ROI绘制工具', 1000, 800)
        
        # 复制图像用于显示
        display_image = self.ff_image.copy()
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        
        # 状态变量
        drawing = False
        roi_type = 'rectangle'  # 'rectangle', 'circle', 'polygon'
        start_point = None
        end_point = None
        center_point = None
        radius = 0
        polygon_points = []  # 存储多边形顶点
        polygon_drawing = False  # 是否正在绘制多边形
        current_roi_mask = np.zeros(self.ff_image.shape, dtype=np.uint8)
        roi_list = []  # 存储多个ROI
        
        def draw_temp_polygon(image, points):
            """绘制临时多边形（正在绘制的多边形）"""
            if len(points) < 2:
                return
            
            # 绘制已添加的点
            for i, point in enumerate(points):
                cv2.circle(image, point, 3, (255, 0, 0), -1)  # 蓝色圆点
                cv2.putText(image, str(i+1), (point[0]+5, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 绘制连接线
            if len(points) > 1:
                points_array = np.array(points, dtype=np.int32)
                cv2.polylines(image, [points_array], False, (255, 0, 0), 2)  # 蓝色线条
            
            # 如果至少3个点，绘制闭合提示线
            if len(points) >= 3:
                cv2.line(image, points[0], points[-1], (0, 255, 255), 1)  # 黄色虚线提示闭合
        
        def update_display():
            nonlocal display_image
            display_image = self.ff_image.copy()
            if len(display_image.shape) == 2:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
            
            # 绘制已保存的ROI
            for i, roi in enumerate(roi_list):
                if roi['type'] == 'rectangle':
                    cv2.rectangle(display_image, roi['start'], roi['end'], (0, 255, 0), 2)
                    cv2.putText(display_image, f'ROI{i+1}', 
                              (roi['start'][0], roi['start'][1]-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif roi['type'] == 'circle':
                    cv2.circle(display_image, roi['center'], roi['radius'], (0, 255, 0), 2)
                    cv2.putText(display_image, f'ROI{i+1}', 
                              (roi['center'][0]-20, roi['center'][1]-roi['radius']-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif roi['type'] == 'polygon':
                    points = np.array(roi['points'], dtype=np.int32)
                    cv2.polylines(display_image, [points], True, (0, 255, 0), 2)
                    # 计算多边形中心点
                    center_x = int(np.mean([p[0] for p in roi['points']]))
                    center_y = int(np.mean([p[1] for p in roi['points']]))
                    cv2.putText(display_image, f'ROI{i+1}', 
                              (center_x-20, center_y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制当前正在绘制的ROI
            if drawing and start_point:
                if roi_type == 'rectangle' and end_point:
                    cv2.rectangle(display_image, start_point, end_point, (0, 0, 255), 2)
                elif roi_type == 'circle' and center_point and radius > 0:
                    cv2.circle(display_image, center_point, radius, (0, 0, 255), 2)
            
            # 绘制临时多边形
            if roi_type == 'polygon' and len(polygon_points) > 0:
                draw_temp_polygon(display_image, polygon_points)
            
            # 添加状态信息
            status_text = f'模式: {roi_type.upper()} | ROI数量: {len(roi_list)} | 操作: r=矩形 c=圆形 p=多边形 s=保存 d=删除 q=计算 Enter=确认 Esc=取消'
            cv2.putText(display_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, end_point, center_point, radius, polygon_points, polygon_drawing, display_image
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if roi_type == 'polygon':
                    # 多边形模式：添加顶点
                    point = (x, y)
                    polygon_points.append(point)
                    
                    # 检查是否应该闭合多边形（至少3个点且点击位置接近第一个点）
                    if len(polygon_points) >= 3:
                        first_point = polygon_points[0]
                        distance = np.sqrt((point[0] - first_point[0])**2 + (point[1] - first_point[1])**2)
                        if distance < 20:  # 20像素范围内认为点击了第一个点
                            # 闭合多边形
                            polygon_drawing = True
                            # 保存多边形ROI
                            if len(polygon_points) >= 3:
                                roi = {
                                    'type': 'polygon',
                                    'points': polygon_points.copy()
                                }
                                roi_list.append(roi)
                                print(f'保存多边形ROI，当前ROI数量: {len(roi_list)}')
                                polygon_points = []
                                polygon_drawing = False
                            update_display()
                            return
                    
                    # 更新显示
                    update_display()
                else:
                    # 矩形和圆形模式
                    drawing = True
                    start_point = (x, y)
                    end_point = (x, y)
                    center_point = (x, y)
                    radius = 0
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing and roi_type != 'polygon':
                    if roi_type == 'rectangle':
                        end_point = (x, y)
                    elif roi_type == 'circle':
                        center_point = (x, y)
                        radius = int(np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2))
                    update_display()
                    
            elif event == cv2.EVENT_LBUTTONUP:
                if roi_type != 'polygon':
                    drawing = False
                    if roi_type == 'rectangle':
                        end_point = (x, y)
                    elif roi_type == 'circle':
                        center_point = (x, y)
                        radius = int(np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2))
                    update_display()
        
        cv2.setMouseCallback('FF Image - ROI绘制工具', mouse_callback)
        
        # 初始化显示
        update_display()
        
        while True:
            cv2.imshow('FF Image - ROI绘制工具', display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # 切换到矩形模式
                roi_type = 'rectangle'
                polygon_points = []
                polygon_drawing = False
                print("切换到矩形模式")
                update_display()
                
            elif key == ord('c'):  # 切换到圆形模式
                roi_type = 'circle'
                polygon_points = []
                polygon_drawing = False
                print("切换到圆形模式")
                update_display()
                
            elif key == ord('p'):  # 切换到多边形模式
                roi_type = 'polygon'
                polygon_points = []
                polygon_drawing = False
                print("切换到多边形模式")
                update_display()
                
            elif key == ord('d'):  # 删除最后一个ROI
                if roi_list:
                    roi_list.pop()
                    print(f'删除最后一个ROI，当前ROI数量: {len(roi_list)}')
                    update_display()
                else:
                    print("没有ROI可删除")
                    
            elif key == ord('s'):  # 保存当前ROI
                if roi_type == 'rectangle' and start_point and end_point:
                    roi_list.append({
                        'type': 'rectangle',
                        'start': start_point,
                        'end': end_point
                    })
                    print(f'保存矩形ROI，当前ROI数量: {len(roi_list)}')
                    start_point = end_point = None
                    update_display()
                elif roi_type == 'circle' and center_point and radius > 0:
                    roi_list.append({
                        'type': 'circle',
                        'center': center_point,
                        'radius': radius
                    })
                    print(f'保存圆形ROI，当前ROI数量: {len(roi_list)}')
                    center_point = None
                    radius = 0
                    update_display()
                elif roi_type == 'polygon' and len(polygon_points) >= 3:
                    roi = {
                        'type': 'polygon',
                        'points': polygon_points.copy()
                    }
                    roi_list.append(roi)
                    print(f'保存多边形ROI，当前ROI数量: {len(roi_list)}')
                    polygon_points = []
                    polygon_drawing = False
                    update_display()
                else:
                    print("请先绘制一个ROI")
                    
            elif key == ord('q'):  # 实时计算当前ROI
                if roi_list:
                    # 临时合并当前ROI
                    temp_roi_mask = np.zeros(self.ff_image.shape, dtype=np.uint8)
                    for roi in roi_list:
                        if roi['type'] == 'rectangle':
                            x1, y1 = roi['start']
                            x2, y2 = roi['end']
                            temp_roi_mask[y1:y2, x1:x2] = 255
                        elif roi['type'] == 'circle':
                            cv2.circle(temp_roi_mask, roi['center'], roi['radius'], 255, -1)
                        elif roi['type'] == 'polygon':
                            points = np.array(roi['points'], dtype=np.int32)
                            cv2.fillPoly(temp_roi_mask, [points], 255)
                    
                    # 临时设置ROI掩码并计算
                    original_roi_mask = self.roi_mask
                    self.roi_mask = temp_roi_mask
                    
                    try:
                        result = self.calculate_fat_fraction()
                        self.display_results_on_image(display_image, result)
                        print("✅ 实时计算完成！按任意键继续...")
                        cv2.waitKey(0)
                    except Exception as e:
                        print(f'❌ 计算失败: {str(e)}')
                    finally:
                        # 恢复原始ROI掩码
                        self.roi_mask = original_roi_mask
                else:
                    print("请先保存至少一个ROI")
                    
            elif key == 13:  # Enter键 - 确认并计算
                if roi_list:
                    # 合并所有ROI
                    self.roi_mask = np.zeros(self.ff_image.shape, dtype=np.uint8)
                    for roi in roi_list:
                        if roi['type'] == 'rectangle':
                            x1, y1 = roi['start']
                            x2, y2 = roi['end']
                            self.roi_mask[y1:y2, x1:x2] = 255
                        elif roi['type'] == 'circle':
                            cv2.circle(self.roi_mask, roi['center'], roi['radius'], 255, -1)
                        elif roi['type'] == 'polygon':
                            points = np.array(roi['points'], dtype=np.int32)
                            cv2.fillPoly(self.roi_mask, [points], 255)
                    
                    print(f'✅ 确认选择，共{len(roi_list)}个ROI区域')
                    
                    # 实时计算并显示结果
                    try:
                        # 计算脂肪分数
                        result = self.calculate_fat_fraction()
                        
                        # 在图像上显示计算结果
                        self.display_results_on_image(display_image, result)
                        
                        print("✅ 实时计算完成！按任意键继续...")
                        cv2.waitKey(0)
                        
                    except Exception as e:
                        print(f'❌ 计算失败: {str(e)}')
                    
                    break
                else:
                    print("请先绘制并保存至少一个ROI")
                    
            elif key == 27:  # Esc键
                print("取消选择")
                break
        
        cv2.destroyAllWindows()
        
        if self.roi_mask is None:
            raise ValueError("未选择ROI区域")
    
    def display_results_on_image(self, image, result):
        """在图像上显示计算结果"""
        # 创建结果图像
        result_image = image.copy()
        
        # 计算ROI边界框
        roi_binary = (self.roi_mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 绘制ROI轮廓
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 3)
            
            # 计算ROI边界框
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # 在ROI附近显示结果
            text_x = max(10, x)
            text_y = max(50, y - 20)
            
            # 创建半透明背景
            overlay = result_image.copy()
            cv2.rectangle(overlay, (text_x-5, text_y-35), (text_x+400, text_y+10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
            
            # 显示主要结果
            main_text = f'平均脂肪分数: {result["mean_fat_fraction"]:.3f}'
            cv2.putText(result_image, main_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示详细信息
            detail_texts = [
                f'标准差: {result["std_fat_fraction"]:.3f}',
                f'中位数: {result["median_fat_fraction"]:.3f}',
                f'像素数量: {result["pixel_count"]}',
                f'ROI面积: {result["roi_area"]}'
            ]
            
            for i, text in enumerate(detail_texts):
                cv2.putText(result_image, text, (text_x, text_y + 25 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示结果图像
        cv2.imshow('FF Image - 计算结果', result_image)
        
        return result_image
    
    def calculate_fat_fraction(self):
        """计算ROI区域的脂肪分数值"""
        if self.roi_mask is None:
            raise ValueError("ROI掩码未定义，请先加载或创建ROI掩码")
        
        print("正在计算脂肪分数值...")
        
        # 检测图像像素值范围并决定是否需要归一化
        min_pixel = np.min(self.ff_image)
        max_pixel = np.max(self.ff_image)
        print(f'图像像素值范围: {min_pixel} - {max_pixel}')
        
        # 判断是否需要归一化
        if max_pixel <= 100 and min_pixel >= 0:
            print("检测到像素值范围在0-100之间，直接使用原始像素值作为脂肪分数")
            normalized_ff_image = self.ff_image.astype(np.float32)
        else:
            print(f'像素值范围超出0-100，使用最大值{max_pixel}进行归一化')
            # 使用最大值进行归一化到0-1范围
            normalized_ff_image = self.ff_image.astype(np.float32) / max_pixel
            print(f'归一化后范围: {np.min(normalized_ff_image):.3f} - {np.max(normalized_ff_image):.3f}')
        
        # 确保ROI掩码是二值的
        roi_binary = (self.roi_mask > 0).astype(np.uint8)
        
        # 提取ROI区域内的FF值
        roi_ff_values = normalized_ff_image[roi_binary == 1]
        
        if len(roi_ff_values) == 0:
            raise ValueError("ROI区域内没有有效的像素值")
        
        # 计算统计信息
        self.fat_fraction_values = roi_ff_values
        
        # 基本统计
        mean_ff = np.mean(roi_ff_values)
        std_ff = np.std(roi_ff_values)
        median_ff = np.median(roi_ff_values)
        min_ff = np.min(roi_ff_values)
        max_ff = np.max(roi_ff_values)
        
        # 计算像素数量
        pixel_count = len(roi_ff_values)
        roi_area = np.sum(roi_binary)
        
        # 保存结果
        self.results = {
            'mean_fat_fraction': mean_ff,
            'std_fat_fraction': std_ff,
            'median_fat_fraction': median_ff,
            'min_fat_fraction': min_ff,
            'max_fat_fraction': max_ff,
            'pixel_count': pixel_count,
            'roi_area': roi_area,
            'roi_coverage_percent': (roi_area / (self.ff_image.shape[0] * self.ff_image.shape[1])) * 100,
            'original_pixel_range': {'min': int(min_pixel), 'max': int(max_pixel)},
            'normalization_applied': not (max_pixel <= 100 and min_pixel >= 0),
            'normalization_factor': max_pixel if not (max_pixel <= 100 and min_pixel >= 0) else 1.0
        }
        
        print(f"脂肪分数计算完成:")
        print(f'  平均脂肪分数: {mean_ff:.3f}')
        print(f'  标准差: {std_ff:.3f}')
        print(f'  中位数: {median_ff:.3f}')
        print(f'  最小值: {min_ff:.3f}')
        print(f'  最大值: {max_ff:.3f}')
        print(f'  像素数量: {pixel_count}')
        print(f'  ROI面积: {roi_area}')
        print(f'  覆盖率: {self.results["roi_coverage_percent"]:.2f}%')
        print(f'  归一化: {"是" if self.results["normalization_applied"] else "否"}')
        
        return self.results
    
    def visualize_results(self, save_path=None):
        """可视化分析结果"""
        if self.fat_fraction_values is None:
            raise ValueError("请先计算脂肪分数值")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 原始FF图像
        axes[0, 0].imshow(self.ff_image, cmap='gray')
        axes[0, 0].set_title('原始FF图像', fontsize=14)
        axes[0, 0].axis('off')
        
        # 2. ROI掩码叠加
        axes[0, 1].imshow(self.ff_image, cmap='gray')
        roi_overlay = np.ma.masked_where(self.roi_mask == 0, self.roi_mask)
        axes[0, 1].imshow(roi_overlay, cmap='Reds', alpha=0.5)
        axes[0, 1].set_title('ROI区域叠加', fontsize=14)
        axes[0, 1].axis('off')
        
        # 3. 脂肪分数值分布直方图
        axes[1, 0].hist(self.fat_fraction_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(self.results['mean_fat_fraction'], color='red', linestyle='--', 
                          label=f'平均值: {self.results["mean_fat_fraction"]:.3f}')
        axes[1, 0].axvline(self.results['median_fat_fraction'], color='green', linestyle='--', 
                          label=f'中位数: {self.results["median_fat_fraction"]:.3f}')
        axes[1, 0].set_xlabel('脂肪分数值')
        axes[1, 0].set_ylabel('像素数量')
        axes[1, 0].set_title('脂肪分数值分布直方图', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 统计信息表格
        normalization_info = "是" if self.results['normalization_applied'] else "否"
        norm_factor = self.results['normalization_factor']
        original_range = f"{self.results['original_pixel_range']['min']}-{self.results['original_pixel_range']['max']}"
        
        stats_data = [
            ['平均脂肪分数', f'{self.results["mean_fat_fraction"]:.3f}'],
            ['标准差', f'{self.results["std_fat_fraction"]:.3f}'],
            ['中位数', f'{self.results["median_fat_fraction"]:.3f}'],
            ['最小值', f'{self.results["min_fat_fraction"]:.3f}'],
            ['最大值', f'{self.results["max_fat_fraction"]:.3f}'],
            ['像素数量', f'{self.results["pixel_count"]}'],
            ['ROI面积', f'{self.results["roi_area"]}'],
            ['覆盖率', f'{self.results["roi_coverage_percent"]:.2f}%'],
            ['原始像素范围', original_range],
            ['是否归一化', normalization_info],
            ['归一化因子', f'{norm_factor:.1f}' if norm_factor != 1.0 else "无"]
        ]
        
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=stats_data, 
                                colLabels=['统计指标', '数值'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes[1, 1].set_title('统计结果', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'可视化结果已保存: {save_path}')
        
        plt.show()
    
    def save_results(self, output_path):
        """保存分析结果到文件"""
        if not self.results:
            raise ValueError("没有分析结果可保存")
        
        # 转换NumPy类型为Python原生类型
        def convert_numpy_types(obj):
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
        
        # 保存为JSON格式
        json_path = output_path.replace('.json', '_results.json')
        results_converted = convert_numpy_types(self.results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV格式
        csv_path = output_path.replace('.json', '_fat_fraction_values.csv')
        df = pd.DataFrame({
            'pixel_index': range(len(self.fat_fraction_values)),
            'fat_fraction_value': self.fat_fraction_values
        })
        df.to_csv(csv_path, index=False)
        
        print(f"分析结果已保存:")
        print(f"JSON文件: {json_path}")
        print(f"CSV文件: {csv_path}")
        
        return json_path, csv_path

def analyze_ff_image(ff_image_path, roi_mask_path=None, output_dir=None):
    """
    分析FF图像的脂肪分数值
    
    Args:
        ff_image_path: FF图像文件路径
        roi_mask_path: ROI掩码文件路径（可选）
        output_dir: 输出目录（可选）
    
    Returns:
        dict: 分析结果
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(ff_image_path), "ff_analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分析器
    analyzer = FFImageAnalyzer(ff_image_path, roi_mask_path)
    
    try:
        # 加载图像
        analyzer.load_images()
        
        # 如果没有ROI掩码，则手动创建
        if analyzer.roi_mask is None:
            analyzer.create_manual_roi()
        
        # 计算脂肪分数
        results = analyzer.calculate_fat_fraction()
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(ff_image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        analyzer.save_results(output_path)
        
        # 生成可视化结果
        viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        analyzer.visualize_results(viz_path)
        
        return results
        
    except Exception as e:
        logger.error(f"分析FF图像失败: {str(e)}")
        raise

def batch_analyze_ff_images(ff_images_dir, roi_masks_dir=None, output_dir=None):
    """
    批量分析FF图像
    
    Args:
        ff_images_dir: FF图像目录
        roi_masks_dir: ROI掩码目录（可选）
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.join(ff_images_dir, "batch_ff_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有FF图像文件
    ff_files = [f for f in os.listdir(ff_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    # 使用自然排序，正确处理数字排序
    ff_files.sort(key=natural_sort_key)
    
    print(f'找到 {len(ff_files)} 个FF图像文件')
    
    all_results = []
    
    for i, ff_file in enumerate(ff_files):
        print(f'\n处理第 {i+1}/{len(ff_files)} 个文件: {ff_file}')
        
        ff_path = os.path.join(ff_images_dir, ff_file)
        
        # 查找对应的ROI掩码
        roi_path = None
        if roi_masks_dir and os.path.exists(roi_masks_dir):
            roi_file = ff_file.replace('.png', '_roi.png').replace('.jpg', '_roi.png')
            roi_path = os.path.join(roi_masks_dir, roi_file)
            if not os.path.exists(roi_path):
                roi_path = None
        
        # 如果没有ROI掩码，跳过此图像
        if roi_path is None:
            print(f"跳过 {ff_file}：未找到对应的ROI掩码")
            continue
        
        # 分析单个图像
        result = analyze_ff_image(ff_path, roi_path, output_dir)
        
        if result:
            result['image_file'] = ff_file
            all_results.append(result)
    
    # 保存批量分析结果
    if all_results:
        batch_results_path = os.path.join(output_dir, "batch_analysis_summary.json")
        with open(batch_results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 创建汇总表格
        summary_data = []
        for result in all_results:
            summary_data.append({
                '图像文件': result['image_file'],
                '平均脂肪分数': result['mean_fat_fraction'],
                '标准差': result['std_fat_fraction'],
                '中位数': result['median_fat_fraction'],
                '像素数量': result['pixel_count'],
                'ROI面积': result['roi_area'],
                '覆盖率(%)': result['roi_coverage_percent'],
                '原始像素范围': f"{result['original_pixel_range']['min']}-{result['original_pixel_range']['max']}",
                '是否归一化': "是" if result['normalization_applied'] else "否",
                '归一化因子': f"{result['normalization_factor']:.1f}" if result['normalization_factor'] != 1.0 else "无"
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, "batch_analysis_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n批量分析完成！")
        print(f'成功处理: {len(all_results)} 个文件')
        print(f'结果保存到: {output_dir}')
        print(f'汇总文件: {batch_results_path}')
    
    return all_results
