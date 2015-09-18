import numpy as np

def flattenimg(img):
    # reshape to 2-D
    return  img.reshape(-1, img.shape[-1])

def sortimg(img):
    # obtain indices of a sorted matrix
    id = np.lexsort(img.T)
    # rearrange matrix based on this index and return
    return  img[id, :]

""" Returns separate histogram for each color in a pixel: 3 for RGB, 4 for CMYK, 1 for Grayscale
"""
def gethistogram(img):
    # reshape image
    imgr = flattenimg(img)
    # get histogram for each color the numpy way: use this!
    hist = []
    [hist.append(np.histogram(imgr[:,n], bins=256)[0]) for n in xrange(imgr.shape[-1])]

    return np.asarray(hist)

"""
    # horribly slow vanilla implementation: use only for proof of work
    hist = np.zeros((256, imgr.shape[-1]), dtype=np.uint32)
    for pixel in imgr:
        for color in xrange(imgr.shape[-1]):
            hist[pixel.item(color), color] += 1
    return hist
"""

def equalize(img, hist):
    # generate lookup table(s)
    imgsize = img.shape[0]*img.shape[1]
    lut = []

    # if grayscale add an axis
    gs = False
    if len(img.size) < 3:
        gs = True
        img = img.reshape((img.shape[0], img.shape[1], 1))

    # for each color (1 for greyscale, 3 for RGB, 4 for CMYK)
    for color in xrange(img.shape[-1]):
        # generate cumulative distribution function
        cdf = hist[color].cumsum()
        # retrieve first non-zero element of cdf
        cdfmin = 0
        while cdf.item(cdfmin) == 0:
            cdfmin += 1
        # equalize the colors / generate lookup table
        norm = (cdf - cdfmin)*255/(imgsize - cdfmin)
        lut.append(norm)

    lut = np.asarray(lut, dtype=np.uint8)

    # remap colors in img according to lut (lookup table):
    eq = np.dstack([lut[n][img[...,n]] for n in xrange(img.shape[-1])])
    # if grayscale remove added axis
    if gs:
        eq = eq.reshape((img.shape[0], img.shape[1]))
    return eq

def getgrayscale(img):
    # sum colors (elements along last axis) and divide by number
    # of colors
    if img.dtype == "float32":
        img[:] = np.asarray(img*255, dtype=np.uint8)
    return np.asarray(np.sum(img,axis=-1)/img.shape[-1], dtype=np.uint8)

def getunique(imgs):
    # detect unique colors:
    # diff() along image length to detect changes in pixel color
    # any() along axis 0 to count number of color changes
    return np.any(np.diff(imgs, axis=0), axis=1).sum() + 1

def getbackground(img, imgs, thrs=10):
    # count occurences of each unique pixel color
    id = np.append([0], np.any(np.diff(imgs, axis=0), axis=-1).cumsum())
    count = np.bincount(id)
    # calculate index of most common pixel
    idc = 0
    if np.argmax(count) > 0:
        idc = count[:np.argmax(count)].cumsum()[-1]
    # obtain most common pixel
    pxc = imgs[idc]
    # for each pixel in original image obtain its difference to pxc.
    # if the absolute of the difference between ANY color in the two
    # is below a certain threshold return TRUE for that pixel, hence
    # indentify it as a background pixel.
    # else return FALSE (foreground).
    back = np.any(np.abs(img - pxc) <= thrs, axis=-1)

    return back

""" applies otsu's automatic thresholding algorithm to separate
    background and foreground:
    1. Compute histogram and probabilities of intensity levels
    2. Initialize wb and mb
    3. Step through each intensity level and compute:
    4. wb and mb for that level
    5. var**2b for that level
    6. Once found for all levels, take maximum var**2
"""
def otsu(img):
    # compute grayscale version of image
    imgg = getgrayscale(img)
    # compute histogram and probabilities
    hist = np.histogram(imgg,bins=255)[0]
    # compute sum
    sum = np.sum(hist*xrange(255)) 
    # initialize loop variables
    wb = 0
    sumb = 0
    tot = imgg.size
    thr = []

    for n in xrange(255):
        wb += hist[n]
        wf = tot - wb
        if wb == 0:
            continue
        if wf == 0:
            break
        sumb += n*hist[n]
        mb = 1.0*sumb/wb
        mf = 1.0*(sum - sumb)/wf
        thr.append(wb*wf*(mb-mf)**2)

    lvl = np.argmax(thr) + 1
    if lvl < 128:
        return imgg >= lvl
    else:
        return imgg < lvl

def thin(img):
    # binarize image
    imgb = otsu(img)

    # make copy of binarized image to store thinned image
    imgt = np.copy(imgb)
    # obtain image containing only chain code points
    for row in xrange(1, imgb.shape[0] - 1):
        for col in xrange(1, imgb.shape[1] - 1):
            if imgb.item((row,col)):
                imgt[row,col] = np.logical_not(np.all(imgb[row-1:row+2, col-1:col+2]))

    return imgt

def segment(img):
    # attempt to cluster our boundary pixels into separate objects
    imgt = np.copy(img)
    pixels = np.transpose(np.nonzero(imgt))
    paths = []

    while pixels.size > 0:
        path = dfsi(imgt, pixels[0,0], pixels[0,1])
        path = path.reshape((path.size/2, 2))
        paths.append(path)
        pixels = np.transpose(np.nonzero(imgt))
        
    return paths

def dfsi(bitmap, start_row, start_col):
    stack = []
    stack.append(start_row)
    stack.append(start_col)
    path = []
    origin = np.asarray([1,1])

    while len(stack) > 0:
        col = stack.pop()
        row = stack.pop()
        bitmap[row, col] = False
        path = np.append(path, [row, col])

        edges = np.transpose(np.nonzero(bitmap[row-1:row+2, col-1:col+2])) - origin
        for edge in edges:
            stack.append(row+edge[0])
            stack.append(col+edge[1])

    return path
