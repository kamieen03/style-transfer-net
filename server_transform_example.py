from PIL import Image
from StyleTransfer import StyleTransfer
import numpy as np

im = Image.open("data/content/057725.jpg")
im = np.asarray(im.resize((576, 1024))).transpose(2,0,1)
style = Image.open("data/style/2.jpg")
style = np.asarray(style.resize((576, 1024))).transpose(2,0,1)
s = StyleTransfer()
print(im.shape, style.shape)
s.set_style(style, 1.0)
t = s.stylize_frame(im).transpose(1,2,0)
print(t.shape)
Image.fromarray(t).show()
