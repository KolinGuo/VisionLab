from matplotlib import pyplot as plt
from PIL import Image
import os
import numpy as np

def show_mask(mask: np.ndarray, ax, random_color=False) -> None:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box: np.ndarray, ax, label) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                               facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)

input_dir = 'example_imgs_openx'
output_dir = '/home/xuanlin/Downloads/test_openx_det_seg'
os.makedirs(output_dir, exist_ok=True)

os.environ['EFFICIENTVIT_SAM_CKPT_DIR'] = '/home/xuanlin/project_soledad/efficientvit/assets/checkpoints/sam/'
os.environ['EFFICIENTVIT_SAM_TENSORRT_CKPT_DIR'] = '/home/xuanlin/project_soledad/efficientvit/assets/export_models/sam/tensorrt'
os.environ['NANOOWL_TENSORRT_CKPT_DIR'] = '/home/xuanlin/project_soledad/nanoowl/data'

from vision_lab.detection.nanoowl import NanoOwl
detector = NanoOwl(model_variant='google/owlv2-large-patch14-ensemble', 
                   box_threshold=0.2,
                   nms_threshold=0.3,
                   max_image_batch_size=1,
                   device="cuda"
)

from vision_lab.segmentation.efficientvit_sam_tensorrt import EfficientViT_SAM_TensorRT
segmenter = EfficientViT_SAM_TensorRT(model_variant='xl0', 
        max_point_box_batch_num=16,)

prompts = "[eggplant,microwave,banana,fork,yellow towel,red towel,blue towel,red bowl,blue bowl,purple towel,steel bowl,white bowl,red spoon,green spoon,blue spoon,can,strawberry,corn,yellow plate,red plate,cabinet,fridge,screwdriver,mushroom,plastic bottle,green chip bag,brown chip bag,blue chip bag,apple,orange]"
prompts = [x.strip() for x in prompts.strip("][()").split(",")]

for image_path in os.listdir(input_dir):
    image = np.asarray(Image.open(os.path.join(input_dir, image_path)))
    boxes, pred_indices, pred_scores = detector(images=image, prompts=prompts, return_on_cpu=True)
    print(boxes.shape, pred_indices.shape, pred_scores.shape)

    if boxes.shape[0] > 16:
        continue

    if boxes.shape[0] != 0:
        # at least one box detected
        masks, pred_ious = segmenter(image, boxes=boxes, points=None, point_labels=None, return_on_cpu=True)
    else:
        masks, pred_ious = [], []

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.imshow(image)
    for box, pred_idx, pred_score in zip(boxes, pred_indices, pred_scores):
        show_box(box, ax, prompts[pred_idx] + f' {pred_score:.2f}')
    for mask in masks:
        show_mask(mask, ax, random_color=True)
    plt.savefig(os.path.join(output_dir, image_path))
    plt.close()