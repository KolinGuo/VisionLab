from real_robot.utils.logger import get_logger

try:
    from .groundingdino import GroundingDINO
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import GroundingDINO: {e}")

try:
    from .owl_vit import Owl_ViT
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import Owl_ViT: {e}")

try:
    from .nanoowl import NanoOwl
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import NanoOwl: {e}")
    get_logger("VisionLab").warning(
        "First, install NanoOwl via `python3 -m pip install git+https://github.com/xuanlinli17/nanoowl.git`"
        "For TensorRT inference, also ensure that you have the necessary dependencies as in README"
    )
