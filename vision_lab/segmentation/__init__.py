from real_robot.utils.logger import get_logger

try:
    from .sam import SAM
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import SAM: {e}")
    get_logger("VisionLab").warning(
        "Install segment_anything via "
        "`python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git`"
    )

try:
    from .efficientvit_sam import EfficientViT_SAM
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import EfficientViT_SAM (non-TensorRT): {e}")
    get_logger("VisionLab").warning(
        "Install EfficientViT_SAM (non-TensorRT) via "
        "`python3 -m pip install git+https://github.com/xuanlinli17/efficientvit.git`"
        "(You can ignore this if you are using EfficientViT_SAM_TensorRT)"
    )

try:
    from .efficientvit_sam_tensorrt import EfficientViT_SAM_TensorRT
except ImportError as e:
    get_logger("VisionLab").warning(f"Failed to import EfficientViT_SAM_TensorRT: {e}")
    get_logger("VisionLab").warning(
        "For TensorRT inference, ensure that you have the necessary dependencies as in README"
    )

# from .mobile_sam import MobileSAM
