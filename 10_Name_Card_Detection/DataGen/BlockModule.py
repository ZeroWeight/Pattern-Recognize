import os
import numpy as np
import unicodedata
import random
from PIL import Image
from PIL import ImageDraw, ImageFont

def randRGB():
    return (np.random.randint(0, 255),
           np.random.randint(0, 255),
           np.random.randint(0, 255))

def Chinese():
    val = np.random.randint(0x4E00, 0x9FA5)
    return chr(val)

def English():
    # here, we also define some characters are English one
    val = np.random.randint(0x41, 0x7a)
    return chr(val)

def Num():
    val = np.random.randint(0x30, 0x39)
    return chr(val)

def chr_width(c):
    if unicodedata.east_asian_width(c) in ('F','W','A'):
        return 2
    else:
        return 1

generator = [Chinese,English,Num]
fonts = os.listdir('./Fonts')

def BlockModule(mode,revert=0):
    '''
    Return a sub-picture of containing some characters in Chinese, English or num
    :param mode: 0: Chinese, 1: English, 2: Numbers
    :return: img
    '''
    if mode == 0:
        string_len = np.random.randint(3,7)
    else:
        string_len = np.random.randint(5,10)
    height = 32
    weight = 0
    string = ''
    for _ in range(string_len):
        c = generator[mode]()
        string += c
        weight += height * chr_width(c)
    # initialize a random, transparent picture, 0.5 and 1.1 is magic number
    img = Image.new('RGBA', (int(weight * 0.5), int(height*1.2)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_str = random.sample(fonts, 1)[0]
    font = ImageFont.truetype(os.path.join('Fonts',font_str),height)
    if revert == 0:
        draw.text((0,0),string,fill=randRGB(),font=font)
    else:
        draw.text((0, 0), string, fill=randRGB(), font=font)
    #img.save(font_str + '.png')
    return img


if __name__ == '__main__':
    for mode in range(3):
        test_img = BlockModule(mode)
        test_img.save(str(mode)+'.png')
        print (test_img.width,test_img.height)
