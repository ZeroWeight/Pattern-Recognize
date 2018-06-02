# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
__sets = {}

from datasets.name_card import name_card
import numpy as np

for split in ['trainval', 'test']:
  name = 'name_card_real_{}'.format(split)
  __sets[name] = (lambda split=split: name_card(split,'NameCardReal'))

__sets['name_card_fake_train'] = (lambda: name_card('trainval','NameCardFake'))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
