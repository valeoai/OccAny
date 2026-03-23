import os


import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)


# image_path = "/mnt/proj1/eu-25-92/data/nuscenes/imgs/CAM_FRONT/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801733412460.jpg"
image_path = "/lustre/fsn1/projects/rech/kvd/uyl37fq/data/occ3d_nuscenes/imgs/CAM_FRONT/n015-2018-11-21-19-21-35+0800__CAM_FRONT__1542799761912460.jpg"
image = Image.open(image_path)
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)


processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="car")

img0 = Image.open(image_path)
plot_results(img0, inference_state, save_path="test.png")