#!/bin/sh
import sys
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

import util

def main(argv):
    # load image into ndarray
    if len(argv) > 1:
        img = misc.imread(argv[1])
    else:
        img = misc.imread('/home/cilsat/Pictures/1309116686491.jpg')

    # calc sorted 2-D representation of image
    imgs = util.sortimg(util.flattenimg(img))

    # get count of unique colors
    uniq = util.getunique(imgs)
    print uniq

    # get and display background image
    background = util.getbackground(img, imgs)
    displayimg(background)

def displayimg(img):
    plt.imshow(img)
    plt.show()

if __name__ == ('__main__'):
    main(sys.argv)

