import os
from PIL import Image
for filename in ['train.txt']:
  with open(filename,'r') as f:
    for line in f:
      words = line.split(' ')
      index = words[1].split('.')[0].split('/')[1]
      with Image.open(os.path.join('JPEGImages',(line.split(' ')[1].split('/')[1].split('.')[0] + '.jpg'))) as im:
        with open(os.path.join('Annotations',(str(index) + '.xml.bak')),'a') as g:
          if float(words[6]) == 1:
            cls = 'Chinese'
          elif float(words[7]) == 1:
            cls = 'English'
          else:
            cls = 'Number'
          x_max = min(int(words[4]),im.width-1)
          y_max = min(int(words[5]),im.height-1)
          x_min = max(int(words[2]),0)
          y_min = max(int(words[3]),0)
          if (x_max < x_min) or (y_max < y_min):
            continue
          g.write('\t<object>\n\t\t<name>{}</name>\n'.format(cls))
          g.write('\t\t<pose>Unspecified</pose>\n\t\t<truncated>1</truncated>\n\t\t<difficult>0</difficult>\n')
          g.write('\t\t<bndbox>\n\t\t\t<xmin>{}</xmin>\n\t\t\t<ymin>{}</ymin>\n\t\t\t<xmax>{}</xmax>\n\t\t\t<ymax>{}</ymax>\n'.format(
            x_min,y_min,x_max,y_max))
          g.write('\t\t</bndbox>\n\t</object>\n')

for file in os.listdir('Annotations'):
  filename = file.split('.')[0] + '.xml'
  with open(os.path.join('Annotations',filename),'w') as f:
    f.write('<annotation>\n')
    with open(os.path.join('Annotations',file),'r') as g:
      for line in g:
        f.write(line)
    f.write('</annotation>')
