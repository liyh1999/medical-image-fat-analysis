"""
脂肪分数分析软件 - 主程序入口
Fat Fraction Analysis Software v2.1
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 导入并运行主程序
from main import main

if __name__ == "__main__":
    main()
