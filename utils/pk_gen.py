import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def pokemon_gen(prompt, scale=10, n_samples=4):
    pipe = StableDiffusionPipeline.from_pretrained("../my_model", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    disable_safety = False
    if disable_safety:
      def null_safety(images, **kwargs):
          return images, False
      pipe.safety_checker = null_safety

    with autocast("cuda"):
      images = pipe(n_samples*[prompt], guidance_scale=scale).images

    grid = image_grid(images, rows=2, cols=2)
    return grid