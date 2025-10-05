#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向量数据库调试工具主入口
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_db_debugger import main

if __name__ == "__main__":
    main()