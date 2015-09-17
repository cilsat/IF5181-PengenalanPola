#!/bin/python
import sys
import matplotlib.image as mpimg

import util

def main(argv):
    # load image into ndarray
    if len(argv) > 1:
        img = mpimg.imread(argv[1])
    else:
        img = mpimg.imread('/home/cilsat/Pictures/1309116686491.jpg')

    # calc sorted 2-D representation of image
    #imgs = util.sortimg(util.flattenimg(img))

    # get count of unique colors
    #uniq = util.getunique(imgs)
    #print(uniq)

    pltshow(img)
    # get and display histogram(s) of color(s)
    #hist = util.gethistogram(img)
    #plthist(hist)

    # equalize image
    #imgeq = util.equalize(img, hist)
    #pltshow(imgeq)

    # get and display background image with dynamic threshold 15
    #background = util.getbackground(img, imgs, 15)
    #pltshow(background)

    textthin = util.thin(img)
    pltshow(textthin)

def plthist(hist):
    import matplotlib.pyplot as plt
    plt.plot(hist.T)
    plt.show()

def pltshow(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

def pilshow(img):
    from PIL import Image
    import numpy as np
    im = Image.fromarray(np.uint8(img*255))
    im.save('/home/cilsat/Dropbox/kuliah/sem1/pp/test.jpg')

if __name__ == ('__main__'):
    main(sys.argv)

