import numpy as np

def flattenimg(img):
    # reshape to 2-D
    return  img.reshape(-1, img.shape[-1])

def sortimg(img):
    # obtain indexes of a sorted matrix
    id = np.lexsort(img.T)
    # rearrange matrix based on this index and return
    return  img[id, :]

def getunique(imgs):
    # detect unique colors:
    # diff along image length to detect changes in pixel color
    diff = np.diff(imgs, axis=0)
    # any along axis 0 to count number of color changes
    uniq = np.any(diff, axis=1).sum() + 1
    
    return uniq

def getbackground(img, imgs):
    # count occurences of each unique pixel color
    id = np.append([0], np.any(np.diff(imgs, axis=0), axis=-1).cumsum())
    count = np.bincount(id)
    # calculate index of most common pixel
    idc = 0
    if np.argmax(count) > 0:
        idc = count[:np.argmax(count)].cumsum()[-1]
    # obtain most common pixel
    pxc = imgs[idc]
    # compare each pixel in original image with most common pixel
    # return TRUE if ANY of the 3 colors are the same
    back = np.all(np.abs(img - pxc) <= 50, axis=-1)

    return back
