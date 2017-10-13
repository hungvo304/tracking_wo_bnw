# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import frcnn
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import pprint
import numpy as np
import sys

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
#from nets.mobilenet_v1 import mobilenetv1


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


def frcnn_trainval(args):
  """
  args = {'imdb_name':imdb_name,
      'imdbval_name':imdbval_name,
      'max_iters':max_iters,
      'net':network,
      'cfg_file':cfg_file,
      'set_cfgs':set_cfgs,
      'weights':weights,
      'voc_basenet':voc_basenet,
      'tag':tag}
  """

  if args['cfg_file']:
    cfg_from_file(cfg_file)
  if args['set_cfgs']:
    cfg_from_list(set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(args['imdb_name'])
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args['tag'])
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, args['tag'])
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args['imdbval_name'])
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args['net'] == 'vgg16':
    net = vgg16(batch_size=cfg.TRAIN.IMS_PER_BATCH)
  elif args['net'] == 'res50':
    net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=50)
  elif args['net'] == 'res101':
    net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=101)
  elif args['net'] == 'res152':
    net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=152)
  elif args['net'] == 'mobile':
    net = mobilenetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH)
  else:
    raise NotImplementedError

    
  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=args['weights'],
            max_iters=args['max_iters'], basenet=args['basenet'])