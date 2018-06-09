import os
from PIL import Image
from PIL.ImageEnhance import Color, Brightness, Contrast, Sharpness

index_prefix = 0
for color_factor in [0.5,0.8,1.2,1.5]:
  for bright_factor in [0.5,0.8,1.2,1.5]:
    for contrast_factor in [0.5,0.8,1.2,1.5]:
      for sharp_factor in [0.2,0.5,2.0,3.0]:
        index_prefix += 1
        for ori_name in os.listdir(os.path.join('..','NameCardReal','JPEGImages')):
          with Image.open(os.path.join('..','NameCardReal','JPEGImages',ori_name)) as im:
            new_name = str(index_prefix) + ori_name
            im = Color(im).enhance(color_factor)
            im = Brightness(im).enhance(bright_factor)
            im = Contrast(im).enhance(contrast_factor)
            im = Sharpness(im).enhance(sharp_factor)
            im.save(os.path.join('JPEGImages',new_name))
        print (color_factor,bright_factor,contrast_factor,sharp_factor)
            
        
