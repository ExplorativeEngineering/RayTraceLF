#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2020 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
from videofig.videofig import videofig

NUM_IMAGES = 10
PLAY_FPS = 1  # set a large FPS (e.g. 100) to test the fastest speed our script can achieve
SAVE_PLOTS = False  # whether to save the plots in a directory

# Preload images and boxes
imgs = []
def addImg(img):
    imgs.append(img)

for n in range(NUM_IMAGES):
    img = np.random.rand(100,100)
    addImg(img)

def redraw_fn(f, ax):
    img = imgs[f]
    if not redraw_fn.initialized:
        redraw_fn.img_handle = ax.imshow(img)
        redraw_fn.last_time = time.time()
        redraw_fn.text_handle = ax.text(0., 1 - 0.05,
                                        'Resolution {}x{}, FPS: {:.2f}'.format(img.shape[1], img.shape[0], 0),
                                        transform=ax.transAxes,
                                        color='yellow', size=12)
        redraw_fn.initialized = True
    else:
        redraw_fn.img_handle.set_array(img)
        current_time = time.time()
        redraw_fn.text_handle.set_text('Resolution {}x{}, FPS: {:.2f}'.format(img.shape[1], img.shape[0],
                                                                              1 / (current_time - redraw_fn.last_time)))
        redraw_fn.last_time = current_time

redraw_fn.initialized = False

if not SAVE_PLOTS:
    videofig(NUM_IMAGES, redraw_fn, play_fps=PLAY_FPS)
else:
    videofig(NUM_IMAGES, redraw_fn, play_fps=PLAY_FPS, save_dir='example2_save')
