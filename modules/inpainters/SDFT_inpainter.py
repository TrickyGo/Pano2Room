import utils.functions as functions
import torch
from .inpainter import Inpainter
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import os

class SDFTInpainter(Inpainter):
    def __init__(self, subset_name=None):
        super().__init__()

        SD_path = "stabilityai/stable-diffusion-2-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_path, torch_dtype=torch.float16, variant="fp16").to("cuda")

        SDFT_path = f"output/SDFT_weights"
        if os.path.exists(SDFT_path):
            pipe.load_lora_weights(SDFT_path)
        self.inpaint_pipe = pipe

    @torch.no_grad()
    def inpaint(self, img, mask): 
        '''
        :param img: B C H W?
        :param mask: 
        :return:
        '''
        inpaint_mask_pil = Image.fromarray(mask.detach().cpu().squeeze(0).squeeze(0).float().numpy() * 255).convert("RGB")
  
        rendered_image_pil = functions.tensor_to_pil(img)

        prompt = ""
        generator = torch.Generator(device="cuda").manual_seed(0)

        inpainted_image_pil = self.inpaint_pipe(
        prompt=prompt,
        image=rendered_image_pil,
        mask_image=inpaint_mask_pil,
        guidance_scale=7.5,
        num_inference_steps=30,  
        generator=generator,
        ).images[0]
        result = functions.pil_to_tensor(inpainted_image_pil)

        return result.to(torch.float32)
