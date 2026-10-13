"""大乐透 Analysis 阶段：可复现的描述性统计与统计检验。"""

__version__ = "0.1.0"

from .config import AnalysisConfig
from .runner import AnalysisRunner

__all__ = ["AnalysisConfig", "AnalysisRunner", "__version__"]
