"""
NIfTI 3D文件查看器模块
提供3D NIfTI文件的查看和切片功能
"""

import numpy as np
from utils import NIBABEL_AVAILABLE, logger

class NIfTI3DViewer:
    """3D NIfTI文件查看器（参考3D Slicer的内存管理策略）"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.nii_img = None
        self.data_shape = None
        self.data_dtype = None
        self.current_view = 'axial'  # axial, sagittal, coronal
        self.current_slice = 0
        
        # 3D Slicer风格的缓存管理
        self.cached_slices = {}  # 缓存已加载的层面
        self.cache_access_order = []  # 访问顺序，用于LRU
        self.max_cache_size = 20  # 增加缓存大小
        self.max_memory_mb = 100  # 最大内存使用量（MB）
        self.current_memory_mb = 0  # 当前内存使用量
        
    def load_header(self):
        """只加载文件头信息，不加载数据"""
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel库未安装，无法处理NIfTI文件")
        
        try:
            import nibabel as nib
            self.nii_img = nib.load(self.file_path)
            self.data_shape = self.nii_img.shape
            self.data_dtype = self.nii_img.get_data_dtype()
            
            logger.info(f"加载NIfTI文件头: {self.file_path}")
            logger.info(f"数据形状: {self.data_shape}, 数据类型: {self.data_dtype}")
            
            # 初始化切片索引
            if len(self.data_shape) == 3:
                self.current_slice = self.data_shape[2] // 2  # 默认中间层
            elif len(self.data_shape) == 2:
                self.current_slice = 0
            else:
                raise ValueError(f"不支持的NIfTI数据维度: {self.data_shape}")
                
        except Exception as e:
            logger.error(f"加载NIfTI文件头失败: {str(e)}")
            raise
    
    def get_slice(self, view='axial', slice_index=None):
        """获取指定视图和层面的2D图像（延迟加载）"""
        if slice_index is None:
            slice_index = self.current_slice
            
        # 检查缓存
        cache_key = f"{view}_{slice_index}"
        if cache_key in self.cached_slices:
            logger.debug(f"从缓存获取层面: {cache_key}")
            self._update_cache_access(cache_key)  # 更新LRU访问顺序
            return self.cached_slices[cache_key]
        
        # 计算实际切片索引
        if len(self.data_shape) == 2:
            # 2D数据直接返回
            slice_data = self.nii_img.get_fdata()
        else:
            # 3D数据按视图切片
            if view == 'axial':
                if slice_index >= self.data_shape[2]:
                    slice_index = self.data_shape[2] - 1
                slice_data = self.nii_img.dataobj[:, :, slice_index]
            elif view == 'sagittal':
                if slice_index >= self.data_shape[0]:
                    slice_index = self.data_shape[0] - 1
                slice_data = self.nii_img.dataobj[slice_index, :, :]
            elif view == 'coronal':
                if slice_index >= self.data_shape[1]:
                    slice_index = self.data_shape[1] - 1
                slice_data = self.nii_img.dataobj[:, slice_index, :]
            else:
                raise ValueError(f"不支持的视图: {view}")
        
        # 转换为numpy数组并处理
        slice_array = np.array(slice_data)
        
        # 转换为8位灰度图像
        if slice_array.dtype != np.uint8:
            img_min = np.min(slice_array)
            img_max = np.max(slice_array)
            if img_max > img_min:
                slice_array = ((slice_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                slice_array = np.zeros_like(slice_array, dtype=np.uint8)
        
        # 3D Slicer风格的缓存管理
        self._add_to_cache(cache_key, slice_array)
        logger.debug(f"加载并缓存层面: {cache_key}, 形状: {slice_array.shape}")
        
        return slice_array
    
    def _add_to_cache(self, cache_key, slice_array):
        """添加切片到缓存（3D Slicer风格）"""
        # 计算内存使用量
        slice_memory_mb = slice_array.nbytes / (1024 * 1024)
        
        # 如果缓存已存在，先移除
        if cache_key in self.cached_slices:
            self._remove_from_cache(cache_key)
        
        # 检查内存限制，必要时清理缓存
        while (self.current_memory_mb + slice_memory_mb > self.max_memory_mb or 
               len(self.cached_slices) >= self.max_cache_size):
            if not self.cached_slices:
                break
            # 移除最久未使用的缓存项
            oldest_key = self.cache_access_order[0]
            self._remove_from_cache(oldest_key)
        
        # 添加新缓存项
        self.cached_slices[cache_key] = slice_array
        self.cache_access_order.append(cache_key)
        self.current_memory_mb += slice_memory_mb
        
        logger.debug(f"缓存管理: 当前内存使用 {self.current_memory_mb:.1f}MB, 缓存项 {len(self.cached_slices)}")
    
    def _remove_from_cache(self, cache_key):
        """从缓存中移除切片"""
        if cache_key in self.cached_slices:
            slice_array = self.cached_slices[cache_key]
            slice_memory_mb = slice_array.nbytes / (1024 * 1024)
            
            del self.cached_slices[cache_key]
            self.cache_access_order.remove(cache_key)
            self.current_memory_mb -= slice_memory_mb
            
            logger.debug(f"移除缓存项: {cache_key}, 释放内存 {slice_memory_mb:.1f}MB")
    
    def _update_cache_access(self, cache_key):
        """更新缓存访问顺序（LRU）"""
        if cache_key in self.cache_access_order:
            self.cache_access_order.remove(cache_key)
        self.cache_access_order.append(cache_key)
    
    def clear_cache(self):
        """清空所有缓存"""
        self.cached_slices.clear()
        self.cache_access_order.clear()
        self.current_memory_mb = 0
        logger.debug("已清空所有缓存")
    
    def get_view_info(self, view):
        """获取指定视图的信息"""
        if len(self.data_shape) == 2:
            return {
                'view_name': '2D',
                'max_slices': 1,
                'current_slice': 0,
                'shape': self.data_shape
            }
        elif len(self.data_shape) == 3:
            if view == 'axial':
                return {
                    'view_name': '轴状面 (Axial)',
                    'max_slices': self.data_shape[2],
                    'current_slice': self.current_slice,
                    'shape': (self.data_shape[0], self.data_shape[1])
                }
            elif view == 'sagittal':
                return {
                    'view_name': '矢状面 (Sagittal)',
                    'max_slices': self.data_shape[0],
                    'current_slice': self.current_slice,
                    'shape': (self.data_shape[1], self.data_shape[2])
                }
            elif view == 'coronal':
                return {
                    'view_name': '冠状面 (Coronal)',
                    'max_slices': self.data_shape[1],
                    'current_slice': self.current_slice,
                    'shape': (self.data_shape[0], self.data_shape[2])
                }
        return None
    
    def set_view(self, view):
        """设置当前视图"""
        if view in ['axial', 'sagittal', 'coronal']:
            self.current_view = view
            # 重置切片索引到中间位置
            view_info = self.get_view_info(view)
            if view_info:
                self.current_slice = view_info['max_slices'] // 2
            logger.info(f"切换到视图: {view}, 切片: {self.current_slice}")
    
    def set_slice(self, slice_index):
        """设置当前切片"""
        view_info = self.get_view_info(self.current_view)
        if view_info:
            self.current_slice = max(0, min(slice_index, view_info['max_slices'] - 1))
            logger.info(f"设置切片: {self.current_slice}")
    
    def clear_cache(self):
        """清空缓存"""
        self.cached_slices.clear()
        logger.info("清空切片缓存")

def load_nifti_image(file_path, slice_index=None, view='axial'):
    """加载NIfTI图像文件（使用3D查看器）"""
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel库未安装，无法处理NIfTI文件")
    
    try:
        # 创建3D查看器
        viewer = NIfTI3DViewer(file_path)
        viewer.load_header()
        
        # 设置视图和切片
        viewer.set_view(view)
        if slice_index is not None:
            viewer.set_slice(slice_index)
        
        # 获取当前切片
        image_2d = viewer.get_slice(view, viewer.current_slice)
        
        # 将查看器存储到全局变量中，供后续使用
        global current_3d_viewer
        current_3d_viewer = viewer
        
        logger.info(f"加载3D NIfTI文件: {file_path}")
        logger.info(f"当前视图: {view}, 切片: {viewer.current_slice}, 形状: {image_2d.shape}")
        
        return image_2d
        
    except Exception as e:
        logger.error(f"加载NIfTI文件失败: {str(e)}")
        raise

# 全局3D查看器实例
current_3d_viewer = None
