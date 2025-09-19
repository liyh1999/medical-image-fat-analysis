"""
主程序入口
勾画计算界面化软件 v2.1
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, logger

def main():
    """主函数"""
    print("=== 勾画计算界面化软件 v2.1 ===")
    print("正在初始化...")
    
    # 设置日志
    setup_logging()
    logger.info("程序启动")
    
    try:
        # 导入并运行主GUI
        from gui_main import main as gui_main
        gui_main()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖模块都已正确安装")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}")
        print(f"程序运行错误: {str(e)}")
        sys.exit(1)
    
    finally:
        logger.info("程序结束")

if __name__ == "__main__":
    main()
