"""Structure Feature 编排。

第一阶段的 structure statistics 已由 rolling.py 直接基于 processed 的结构事实生成。
本模块保留结构职责边界，避免重复计算 processed 已存在的 descriptive fields。
"""

from __future__ import annotations

from .rolling import build_rolling_features

__all__ = ["build_rolling_features"]
