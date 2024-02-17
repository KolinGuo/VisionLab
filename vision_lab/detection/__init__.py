from real_robot.utils.logger import get_logger

try:
    from .groundingdino import GroundingDINO
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import GroundingDINO: {e}")

try:
    from .owl_vit import Owl_ViT
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import Owl_ViT: {e}")
