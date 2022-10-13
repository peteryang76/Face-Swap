from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt

def MakeGaussianPyramid(image, scale, minsize):
  pyramid = [];
  im = Image.open(image)
  mode = im.mode
  return MakeGaussianPyramidHelper(im, scale, minsize, pyramid, mode)

def MakeGaussianPyramidHelper(im, scale, minsize, pyramid, mode) :
  height, width = im.size
  im_array = np.asarray(im, dtype = np.float64)
  pyramid.append(im_array)
  if (height * scale <= minsize and width * scale <= minsize):
    return pyramid;
  im_array2 = im_array.copy()
  output_array = im_array.copy()
  if mode is "L":
    output_array = gaussian_filter(im_array, 1/(2*scale))
  else:
    for i in range(3):
      output_array[:,:,i] = gaussian_filter(im_array2[:,:,i], 1/(2*scale))
  filtered_im = Image.fromarray(output_array.astype('uint8'))
  height, width = filtered_im.size
  nextImage = filtered_im.resize((int(height * scale), int(width * scale)), Image.BICUBIC)
  nextImage = nextImage.convert(mode)
  return MakeGaussianPyramidHelper(nextImage, scale, minsize, pyramid, mode)

def ShowGaussianPyramid(pyramid):
  numPics = len(pyramid)
  image1_array = pyramid[0]
  width = image1_array.shape[1]
  height = 0
  for i in range(numPics):
    height += pyramid[i].shape[0]
  im = Image.fromarray(image1_array.astype('uint8'))
  mode = im.mode
  if mode is "L":
    background = Image.new("L", (width, height), "white")
  else:
    background = Image.new("RGB", (width, height), "white")
  currHeight = 0
  for i in range(numPics):
    image = Image.fromarray(pyramid[i].astype('uint8'))
    background.paste(image, (0, currHeight))
    currHeight += image.size[1]
  background.save("image.png", "PNG")
  return background

def resizeTemplate(template, width):
  it = Image.open(template)
  mode = it.mode
  orig_height, orig_width = it.size
  scale = width/orig_width
  height = orig_height * scale
  it_array = np.asarray(it, dtype = np.float64)
  it_array2 = it_array.copy()
  if mode is "L":
    output_array = gaussian_filter(it_array2, 1/(2*scale))
  filtered_it = Image.fromarray(output_array.astype('uint8'))
  new_template = filtered_it.resize((int(height), int(width)), Image.BICUBIC)
  return new_template

def drawAt(image, col, row, height, width):
  if image.mode is "L":
    image.convert("RGB")
  draw = ImageDraw.Draw(image)
  radiusH = height/2
  radiusW = width/2
  draw.line((col - radiusW, row - radiusH,
             col - radiusW, row + radiusH), fill = "white", width = 2)
  draw.line((col - radiusW, row - radiusH,
             col + radiusW, row - radiusH), fill = "white", width = 2)
  draw.line((col + radiusW, row - radiusH,
             col + radiusW, row + radiusH), fill = "white", width = 2)
  draw.line((col - radiusW, row + radiusH,
             col + radiusW, row + radiusH), fill = "white", width = 2)
  del draw
  return image

def normxcorr2D(image, template):
    """
    Normalized cross-correlation for 2D PIL images

    Inputs:
    ----------------
    template    The template. A PIL image.  Elements cannot all be equal.

    image       The PIL image.

    Output:
    ----------------
    nxcorr      Array of cross-correlation coefficients, in the range
                -1.0 to 1.0.

                Wherever the search space has zero variance under the template,
                normalized cross-correlation is undefined.

    Implemented for CPSC 425 Assignment 3

    Bob Woodham
    January, 2013
    """

    # (one-time) normalization of template
    t = np.asarray(template, dtype=np.float64)
    t = t - np.mean(t)
    norm = math.sqrt(np.sum(np.square(t)))
    t = t / norm

    # create filter to sum values under template
    sum_filter = np.ones(np.shape(t))

    # get image
    a = np.asarray(image, dtype=np.float64)
    #also want squared values
    aa = np.square(a)

    # compute sums of values and sums of values squared under template
    a_sum = signal.correlate2d(a, sum_filter, 'same')
    aa_sum = signal.correlate2d(aa, sum_filter, 'same')
    # Note:  The above two lines could be made more efficient by
    #        exploiting the fact that sum_filter is separable.
    #        Even better would be to take advantage of integral images

    # compute correlation, 't' is normalized, 'a' is not (yet)
    numer = signal.correlate2d(a, t, 'same')
    # (each time) normalization of the window under the template
    denom = np.sqrt(aa_sum - np.square(a_sum)/np.size(t))

    # wherever the denominator is near zero, this must be because the image
    # window is near constant (and therefore the normalized cross correlation
    # is undefined). Set nxcorr to zero in these regions
    tol = np.sqrt(np.finfo(denom.dtype).eps)
    nxcorr = np.where(denom < tol, 0, numer/denom)

    # if any of the coefficients are outside the range [-1 1], they will be
    # unstable to small variance in a or t, so set them to zero to reflect
    # the undefined 0/0 condition
    nxcorr = np.where(np.abs(nxcorr-1.) > np.sqrt(np.finfo(nxcorr.dtype).eps),nxcorr,0)

    return nxcorr



def FindTemplate(pyramid, template, threshold):
  orig_im_array = pyramid[0]
  orig_im_array2 = orig_im_array.copy()
  template_width = 15
  # adjust template size
  new_template = resizeTemplate(template, template_width)
  orig_im = Image.fromarray(pyramid[0].astype('uint8'))
  orig_im2 = orig_im.copy()
  template2 = new_template.copy()
  t_width, t_height = template2.size
  orig_width, orig_height = orig_im.size
  for image_array in pyramid:
    # load image from pyramid
    im = Image.fromarray(image_array.astype('uint8'))
    # calculate ncc array of the image
    ncc_array = ncc.normxcorr2D(im, template2)
    im_width, im_height = im.size
    scale = orig_height / im_height
    nccBin_array = ncc_array > threshold
    # draw at where ncc value > threshold
    for row in range(0, ncc_array.shape[0]):
      for col in range(0, ncc_array.shape[1]):
        if ncc_array[row, col] > threshold:
          newRow = int(row * scale)
          newCol = int(col * scale)
          drawAt(orig_im2, newCol, newRow,
                 t_height * scale, t_width * scale)
  image = Image.fromarray(orig_im_array2, "RGB")
  orig_im2.save("faceDetection.png", "PNG")
  return orig_im2