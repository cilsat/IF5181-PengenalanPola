#!/bin/python
import sys
import matplotlib.image as mpimg

import util

def main(argv):
    # load image into ndarray
    if len(argv) > 1:
        testimg = mpimg.imread(argv[1])
    else:
        testimg = mpimg.imread('/home/cilsat/Dropbox/kuliah/sem1/pp/train/gnb/plat/1e18d1bebafc0f94110972e2dd546db8_gallery.jpg')

    # calc sorted 2-D representation of image
    #imgs = util.sortimg(util.flattenimg(img))

    # get count of unique colors
    #uniq = util.getunique(imgs)
    #print(uniq)

    #pltshow(img)
    # get and display histogram(s) of color(s)
    #hist = util.gethistogram(img)
    #plthist(hist)

    # equalize image
    #imgeq = util.equalize(img, hist)
    #pltshow(imgeq)

    # get and display background image with dynamic threshold 15
    #background = util.getbackground(img, imgs, 15)
    #pltshow(background)

    # OCR
    # training: only need to run ONCE for each font
    # make sure you have an image containing all lower case letters, upper case letters, and digits in that order
    """
    font = "dejavusans-alphanumeric"
    fontimg = mpimg.imread('train/' + font + '.jpg')
    util.train(fontimg, font)

    # testing: this is only around 50% accurate for text of different font
    print util.test(testimg, font)
    """

    # License Plate recognition
    dataset = "plat"
    util.gnb_train(dataset=dataset)

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

