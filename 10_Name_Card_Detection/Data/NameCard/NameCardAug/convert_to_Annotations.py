import os

for filename in ['train.txt', 'valid.txt']:
  with open(filename,'r') as f:
    for line in f:
      words = line.split(' ')
      index = words[1].split('.')[0]
      with open(os.path.join('Annotations',(str(index) + '.xml.bak')),'a') as g:
        if float(words[6]) == 1:
          cls = 'Number'
        elif float(words[7]) == 1:
          cls = 'English'
        else:
          cls = 'Chinese'
        g.write('\t<object>\n\t\t<name>{}</name>\n'.format(cls))
        g.write('\t\t<pose>Unspecified</pose>\n\t\t<truncated>1</truncated>\n\t\t<difficult>0</difficult>\n')
        g.write('\t\t<bndbox>\n\t\t\t<xmin>{}</xmin>\n\t\t\t<ymin>{}</ymin>\n\t\t\t<xmax>{}</xmax>\n\t\t\t<ymax>{}</ymax>\n'.format(
          int(words[2]),int(words[3]),int(words[4]),int(words[5])))
        g.write('\t\t</bndbox>\n\t</object>\n')

for file in os.listdir('Annotations'):
  filename = file.split('.')[0] + '.xml'
  with open(os.path.join('Annotations',filename),'w') as f:
    f.write('<annotation>\n')
    with open(os.path.join('Annotations',file),'r') as g:
      for line in g:
        f.write(line)
    f.write('</annotation>')
