from real_robot.utils.logger import get_logger

try:
    from .sam import SAM
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import SAM: {e}")

# from .mobile_sam import MobileSAM
