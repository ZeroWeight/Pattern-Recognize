from BlockModule import BlockModule, randRGB
from PIL import Image
from PIL import ImageDraw
import numpy as np
import cv2
import os

margin = [20,20]
whole = [640,360]
def box_in(new_box,boxes):
    for box_exist in boxes:
        if point_in([new_box[0],new_box[1]],box_exist) or point_in([new_box[2],new_box[3]],box_exist):
            return True
        if point_in([new_box[0],new_box[3]],box_exist) or point_in([new_box[1],new_box[2]],box_exist):
            return True
        if point_in([box_exist[0],box_exist[1]],new_box) or point_in([box_exist[2],box_exist[3]],new_box):
            return True
        if point_in([box_exist[0],box_exist[3]],new_box) or point_in([box_exist[1],box_exist[2]],new_box):
            return True
    return False

def point_in(point,box):
    return (box[0] <= point[0] <= box[2]) and (box[1] <= point[1] <= box[3])

def gen_img(img):
    pic_count = np.random.randint(10,20)
    boxes = []

    for _ in range(pic_count):
        mode = np.random.randint(0, 3)
        front_img = BlockModule(mode)
        if np.random.randint(0,5) == 0:
            front_img = front_img.transpose(Image.ROTATE_90)
        while True:
            box = [
                np.random.randint(margin[0], whole[0] - margin[0]),
                np.random.randint(margin[1], whole[1] - margin[1])
                ]
            box = box + [
                int(box[0] +  front_img.width * 0.8 + np.random.rand() * 2.2),
                int(box[1] + front_img.height * 0.8 + np.random.rand() * 2.2)
            ]
            if not box_in(box,boxes):
                break

        front_img = front_img.resize((box[2] - box[0], box[3] - box[1]))
        _,_,_,a = front_img.split()
        img.paste(front_img,box,mask=a)
        box.append(mode)
        boxes.append(box)
        crop_box = [whole[0] * 1.5,whole[1] * 1.5,0,0]
        for box_exist in boxes:
            if box_exist[0] < crop_box[0]:
                crop_box[0] = box_exist[0]
            if box_exist[1] < crop_box[1]:
                crop_box[1] = box_exist[1]
            if box_exist[2] > crop_box[2]:
                crop_box[2] = box_exist[2]
            if box_exist[3] > crop_box[3]:
                crop_box[3] = box_exist[3]
    crop_box[0] -= margin[0]
    crop_box[2] += margin[0]
    crop_box[1] -= margin[1]
    crop_box[3] += margin[1]
    img = img.crop(crop_box)
    # add on noise
    draw = ImageDraw.Draw(img)
    for _ in range(10):
        start_point = (np.random.randint(0,img.width),np.random.randint(0,img.height))
        end_point = (np.random.randint(0,img.width),np.random.randint(0,img.height))
        draw.line([start_point, end_point], fill=randRGB(),width=np.random.randint(1,5))
    for _ in range(100):
        point = (np.random.randint(0, img.width), np.random.randint(0, img.height))
        points = (point[0] - 2,point[1] - 2,point[0] + 2,point[1] + 2,)
        draw.point(point,fill=randRGB())
        draw.ellipse(points,fill=randRGB())
    img.save(os.path.join('pic',(str(img_count) + '.png')))
    return boxes

img_count = 0
ad_time = 600
frame_count = 0
if __name__ == '__main__':
    cap = cv2.VideoCapture('./video/yourname.mp4')
    img_count = 0
    with open('output.txt', 'w') as f:
        while cap.isOpened():
            _, frame = cap.read()

            frame_count += 1
            if frame_count < ad_time:
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop_box = [
                np.random.randint(0,image.width - 800),
                300
                ]
            crop_box += [
                crop_box[0] + 800,
                crop_box[1] + 600
            ]
            img = image.crop(crop_box)
            boxes = gen_img(img)

            for box in boxes:
                chn = eng = num = 0
                if box[4] == 0: chn = 1
                if box[4] == 1: eng = 1
                if box[4] == 2: num = 1
                f.write('{} {} {} {} {} {} {} {} {}\n'.format(
                    img_count, os.path.join('pic',(str(img_count) + '.png')),
                    box[0], box[1], box[2], box[3], chn, eng, num
                ))
            img_count += 1
            print('IMG {} generated'.format(img_count))
