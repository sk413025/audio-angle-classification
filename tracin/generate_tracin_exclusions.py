#!/usr/bin/env python
"""
根據TracIn影響力分數生成樣本排除列表

此腳本是向後兼容層，調用 tracin 模組中的實現。
"""

import os
import sys
import argparse

# 確保能夠導入 tracin 模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 導入新模組的功能
from tracin.scripts.generate_exclusions import run_generate_exclusions, parse_args


def main():
    """主函數"""
    # 顯示兼容性警告
    print("注意: 此腳本是為了向後兼容而保留的。建議直接使用 tracin 模組:")
    print("  python -m tracin.scripts.generate_exclusions [參數]")
    print("開始使用 TracIn 模組...\n")
    
    # 呼叫新模組的功能
    run_generate_exclusions()


if __name__ == "__main__":
    main() 