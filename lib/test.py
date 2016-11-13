from PIL import Image, ImageFilter
import glob

# max_w = 0
# max_h = 0
names = glob.glob("data_dir/formula_images/*.png")
im=Image.open(names[0])

im2 = im.crop(im.getbbox())
print im2.size
# if im2.size[0] > max_w:
#     max_w = im2.size[0]
# if im2.size[1] > max_h:
#     max_h = im2.size[1]

im3 = Image.new("L", (1024, 1024), 255)
im3.paste(im2, im2.getbbox())
im3 = im3.filter(ImageFilter.GaussianBlur(radius=1))
im3.show()

# print("max width is %d" % max_w)
# print("max height is %d" % max_h)
