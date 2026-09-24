"""macrocore: 卫生宏展开核心(语法对象、scope 集、来源映射)。"""

from .syntax import Span, Origin, Syntax
from .expander import Expander, ExpandResult, ALGO_VERSION

__all__ = ["Span", "Origin", "Syntax", "Expander", "ExpandResult", "ALGO_VERSION"]
