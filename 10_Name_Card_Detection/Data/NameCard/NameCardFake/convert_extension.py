from PIL import Image
import os
#for pic in os.listdir('JPEGImages'):
##  print(pic)
#  im = Image.open(os.path.join('JPEGImages',pic))
#  im.save(os.path.join('JPEGImages',(pic.split('.')[0] + '.jpg')))

with open('train.txt','r') as f:
  for line in f:
    with Image.open(os.path.join('JPEGImages',(line.split(' ')[1].split('/')[1].split('.')[0] + '.jpg'))) as im:
      if int(line.split(' ')[2]) < 0 or int(line.split(' ')[3])  < 0:
        print (line)
      if int(line.split(' ')[4]) < int(line.split(' ')[2]) or  int(line.split(' ')[5]) < int(line.split(' ')[3]):
        print (line)
      if int(line.split(' ')[4]) >= im.width:
        print(line,im.width)
      if int(line.split(' ')[5]) >= im.height:
        print(line,im.height)
      
