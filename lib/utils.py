import os
import Image
import ImageChops
import numpy as np

class DataLoader(object):
    def __init__(self, data_dir, train):
        self.data_dir = data_dir

    def get_instance(train):
        formula_number = train.split()[0]
        image_id = train.split()[1]

        with open(self.data_dir + "/im2latex_formulas.lst") as f:
            for i, line in enumerate(f):
                if i == self.formula_number:
                    formula = line;

        image = Image.open(self.data_dir + "/forumla_images/" + self.image_id + ".png")
        return image, formula

    def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
