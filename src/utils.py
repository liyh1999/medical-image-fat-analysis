"""
工具函数模块
包含各种辅助函数和通用工具
"""

import os
import re
import logging
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试导入nibabel库用于处理NIfTI文件
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("警告: nibabel库未安装，无法处理NIfTI文件。请运行: pip install nibabel")

def natural_sort_key(text):
    """自然排序键函数，正确处理数字排序"""
    # 将字符串分割成数字和非数字部分
    parts = re.split(r'(\d+)', text)
    # 将数字部分转换为整数，非数字部分保持字符串
    return [int(part) if part.isdigit() else part for part in parts]

def setup_chinese_font():
    """设置支持中文的字体"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            print(f'设置中文字体: {font}')
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("未找到中文字体，使用默认字体")
    
    plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """设置日志配置"""
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 配置日志
    log_filename = f"logs/ff_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 创建logger
    logger = logging.getLogger('FFAnalyzer')
    logger.info("日志系统初始化完成")
    return logger

def debug_log(func):
    """调试日志装饰器"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('FFAnalyzer')
        logger.debug(f"调用函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

# 全局logger实例
logger = setup_logging()
