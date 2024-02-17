from real_robot.utils.logger import get_logger

try:
    from .tapir import TAPIR
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import TAPIR: {e}")
