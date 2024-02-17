from real_robot.utils.logger import get_logger

try:
    from .deaot import DeAoT
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import DeAoT: {e}")

try:
    from .xmem import XMem
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import XMem: {e}")
