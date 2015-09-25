import numpy as np
import pandas as pd

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
    if len(img.shape) < 3:
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

def binarize(img):
    return getgrayscale(img) < 128

"""
Testing procedure
1. Load training data
2. Process input test data: obtain features of each object we want to identify
3. Predict each object using a specified method
"""
def test(img, setname="sans"):
    # load training features
    trainfeats = np.fromfile('train/' + setname + '.free')
    with open('train/' + setname + '.meta', 'r') as f:
        featdimensions = int(f.read())
    with open('train/order', 'r') as f:
        chars = f.read().replace('\n', '')

    print chars

    trainfeats = trainfeats.reshape((trainfeats.size/(8*featdimensions**2), featdimensions, featdimensions, 8))

    # process test image
    testthin = thin(img)
    objlist = segment(testthin)
    cleanobjlist = preprocess(objlist)

    # attempt to output a letter for each object found in test image
    output = ""
    for obj in cleanobjlist:
        # extract features
        objfeat = freeman(obj, testthin, featdimensions, featdimensions)
        # calculate squared difference against each training feature
        sqdiff = np.sum(np.sum(np.sum(np.square(trainfeats - objfeat), axis=-1), axis=-1), axis=-1)
        output += chars[np.argmin(sqdiff)]

    return output
    
"""
Training procedure
Input an image. The image MUST CONTAIN all lower case letters, all upper case letters, and all digits in that order
The procedure then detects objects and assigns them letters/digits based on this order
"""
def train(img, setname="sans", featdimensions=10):
    thinned = thin(img)
    objlist = segment(thinned)
    cleanobjlist = preprocess(objlist)
    
    # assign letters to features
    with open('train/order', 'r') as f:
        order = f.read().replace('\n', '')

    # get full features: I WANT EM ALL
    feats = []
    [feats.append(freeman(obj, thinned, featdimensions, featdimensions)) for obj in cleanobjlist]
    features = np.vstack([feats])

    # write to external file so we don't need to retrain each time we attempt to recognize a font
    features.tofile('train/' + setname + '.free')
    with open('train/' + setname + '.meta', 'w') as f:
        f.write(str(featdimensions))

"""
Pre-processing
We try to clean the objects as much as possible:
1. Attempt to order the objects as a human would read them
2. Merge certain objects: the dots in 'i' and 'j', the "holes" in 'a', 'o', etc
"""
def preprocess(objlist):
    # get pre-features: we only need absolute centers and heights
    feats = []
    for obj in objlist:
        obj = np.array(obj)
        feats.append([obj[:,0].mean(), obj[:,1].mean(), obj[:,0].max() - obj[:,0].min()])

    prefeats = np.array(feats)

    # figure out where lines change:
    # calc vertical differences between object centers
    # if vertical difference is above a certain threshold (in this case the mean height of objects)
    # we can be reasonably sure that we've changed lines
    # return the indices of new lines
    newlines = np.argwhere(np.diff(prefeats[:,0]) > np.mean(prefeats[:,2])).reshape(-1) + 1

    # obtain indices that would sort objects left to right for each newline
    lines = np.split(prefeats, newlines)
    objid = np.concatenate([np.argsort(lines[nline].T, axis=1)[1] + np.append(newlines, 0)[nline-1] for nline in xrange(len(lines))]).tolist()
    # sort objects
    orderedobjlist = [objlist[ind] for ind in objid]
    # sort prefeats
    prefeats[:] = prefeats[objid]

    # we try to merge objects if they're sufficiently close (for example the dots in 'i' and 'j')
    # here we merge objects that are 10x closer to each other than is expected
    # 10x is completely arbitrary
    delta = np.abs(np.diff(prefeats[:,1]))
    mergeat = np.argwhere(delta < delta.mean()*0.1).reshape(-1).tolist()

    # add "satellite" object to its "planet" object
    for ind in mergeat:
        orderedobjlist[ind] += orderedobjlist[ind+1]

    # remove "satellite" objects from object list
    mergedobjlist = [orderedobjlist[ind] for ind in xrange(len(orderedobjlist)) if (ind-1) not in mergeat]

    return mergedobjlist

"""
Thinning
Discard all pixels except for border pixels
"""
def thin(img):
    # binarize image
    imgb = binarize(img)

    # copy binary image to new one that will contain our final image
    imgt = np.copy(imgb)

    # obtain image containing only border pixels:
    # for each (nonzero) pixel in the original image, check its neighbours
    # if NOT ALL of its neigbours are (nonzero) pixels, then it's a border pixel
    # if ALL of its neighbouts are (nonzero) pixels, then discard it
    # as we are setting pixels to zero, we need to output it to a new array: otherwise, our loop will detect false positives whenever it moves on to the next pixel
    """
    for row in xrange(1, imgb.shape[0] - 1):
        for col in xrange(1, imgb.shape[1] - 1):
            if imgb.item(row,col):
                imgt.itemset((row,col), np.logical_not(np.all(imgb[row-1:row+2, col-1:col+2])))
    """

    [imgt.itemset((row, col), np.logical_not(np.all(imgb[row-1:row+2, col-1:col+2]))) for row in xrange(1, imgb.shape[0] - 1) for col in xrange(1, imgb.shape[1] - 1) if imgb.item(row, col)]

    return imgt

"""
Object detection
Attempt to cluster neighbouring pixels into separate objects
SERIOUSLY NEEDS OPTIMIZATION!
"""
def segment(img):
    # use copy of img as we'll be eliminating elements
    imgt = np.copy(img)
    # obtain positions of border nonzeropix (non-zero elements)
    nonzeropix = np.argwhere(imgt)
    allobjpix = []

    # for all nonzero pixels: for all objects
    while nonzeropix.size > 0:
        # when we encounter a nonzero pixel we try to trace all nonzero pixels connected to it using depth
        # first search. in addition, this procedure erases pixels from the image once they're checked.
        # the search returns a list of connected pixel indices: an object
        obj = dfsi(imgt, nonzeropix)
        # add object path to our output array
        allobjpix += [obj]

        # each time an object is identified we reevaluate the indices of nonzero pixels in our image.
        # this is an expensive operation and should ideally be trashed: alternatively, we should delete 
        # elements from nonzeropix directly once they've been accounted for in our dfs procedure
        nonzeropix = np.argwhere(imgt)
        
    return allobjpix

"""
Iterative depth-first search
"""
def dfsi(bitmap, bitmapnonzero):
    startpixel = [bitmapnonzero.item(0, 0), bitmapnonzero.item(0, 1)]
    
    stack = [startpixel]
    objpix = [startpixel]

    while stack:
        row, col = stack.pop()
        bitmap.itemset((row, col), 0)
        edges = np.argwhere(bitmap[row-1:row+2, col-1:col+2]) - [1, 1]

        for edge in edges:
            nextpixel = [row+edge[0], col+edge[1]]
            if nextpixel not in objpix:
                stack += [nextpixel]
                objpix += [nextpixel]

    return objpix

"""
Object image
Generates image matrix of specified object from pixel positions of object
FOR TESTING ONLY
"""
def getobjimg(objpix):
    objpix = np.asarray(objpix)
    if objpix.min() != 0:
        objpix[:] -= [objpix[...,0].min(), objpix[...,1].min()]

    objimg = np.zeros((objpix[...,0].max()+1, objpix[...,1].max()+1), dtype = int)

    for pix in objpix:
        objimg[pix[0], pix[1]] = True

    import matplotlib.pyplot as plt
    plt.imshow(objimg)
    plt.show()

"""
Extract Features according to UCI set
"""
def uci(objpix):
    # convert to numpy array
    objpix = np.array(objpix)

    y = objpix[...,0]
    x = objpix[...,1]

    ycent = y.mean()
    xcent = x.mean()

    # normalize
    if y.min() != 0 and x.min() != 0:
        objpix[:] -= [y.min(), x.min()]

    y = objpix[...,0]
    x = objpix[...,1]

    height = y.max()
    width = x.max()

    if height == 0 or width == 0: return [None, None, None, None]

    # calc ymean and xmean
    ymean = y.mean()/height
    xmean = x.mean()/width

    # calc yvar and xvar
    yvar = y.var()/height
    xvar = x.var()/width

    return np.array([ymean, xmean, yvar, xvar])

"""
Freeman chain encoding
source: http://www.codeproject.com/Articles/160868/A-C-Project-in-Optical-Character-Recognition-OCR-U
"""
def freeman(objpixels, img, ntracks=5, nsectors=6):
    # copy image so we don't end up overwriting it
    # (this isn't actually necessary)
    objimg = np.copy(img)

    # convert to numpy array
    objpix = np.array(objpixels)

    # get coordinates of center of mass of object
    ycent = objpix[...,0].mean()
    xcent = objpix[...,1].mean()

    # generate two arrays containing the distance and angle of each pixel relative to center of mass
    # get array of positions of all pixels relative to center of mass
    # this is a separate step cos we'll be reusing the position array
    position = objpix - [ycent, xcent]

    # get array of distances: distance is calculated through pythagoras' formula for triangles
    distance = np.sqrt(np.sum(np.square(position), axis=-1))

    # get array of angles: angle is obtained by calculating the inverse tanget of the y position against the x position
    # we need to consider the possibility of negative positions, hence we use the arctan2 numpy function
    # the output of arctan2 is [-pi, pi], so we divide this by pi to get the range [-1, 1]
    angles = np.arctan2(position[...,0], position[...,1])/np.pi

    # now we virtually "divide" our object into separate tracks and sectors
    # pixels are separated into different tracks based on their distance from the center
    # pixels are separated into different sectors based on the angle relative to the center
    distmax = distance.max()
    ftrack = distmax/ntracks
    rtracks = np.linspace(ftrack, distmax, ntracks)
    fsector = 2.0/nsectors - 1
    rsectors = np.linspace(fsector, 1, nsectors)

    chaincode = np.zeros((ntracks, nsectors, 8), dtype=int)
    npix = objpix.shape[0]
    for n in xrange(npix):
        track = np.argwhere(distance[n] <= rtracks)[0]
        sector = np.argwhere(angles[n] <= rsectors)[0]
        row = objpix.item(n, 0)
        col = objpix.item(n, 1)
        # find neighbouring pixels, and figure out which of them are nonzero
        relation = np.argwhere(np.delete(objimg[row-1:row+2, col-1:col+2].reshape(-1), 4))[:,0]
        for rel in relation:
            chaincode[track, sector, rel] += 1

    pernpix = 1.0/npix

    return chaincode*pernpix

"""
Gaussian Naive Bayes: continuous version of Naive Bayes classifier
Recall that Naive Bayes states that P(A|B) = P(A)*P(B|A)/P(B)
For the continuous version, each random variable (feature) is assumed to be independent and distributed normally

A "feature" is equvalent to an "attribute" or "random variable" in this case. For freeman chain encoding,
a feature is the value contained in a single field of our track x sector x direction matrix.

"Features" refers to the collection of features for a single instance of an output class, for instance the
features of the output class "z" for a font "dejavusans".

A collection of all the features for all output classes is a "feature set", 

We define a "training set" to be a collection of feature sets with each output class represented equally.
"""
def gnb_train(feats, chars):
    pdfeats = pd.Series(feats, index=[char for char in chars])

