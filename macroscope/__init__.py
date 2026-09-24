from .analyzer import analyze_source
from .storage import Storage

ALGORITHM_VERSION = "macroscope-1.0.0"
COORDINATE_BASIS = "unicode-code-point-offset-0,line-1,column-0"

__all__ = ["ALGORITHM_VERSION", "COORDINATE_BASIS", "Storage", "analyze_source"]
