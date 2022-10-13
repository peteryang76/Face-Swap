from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
from scipy.ndimage import gaussian_filter
import cv2
import SetUpImages
# import matplotlib.pyplot as plt

TEMPLATE_SIZE = 15

def makeGaussianPyramid(image, scale, minsize):
  pyramid = []
  im = Image.open(image)
  mode = im.mode
  return makeGaussianPyramidHelper(im, scale, minsize, pyramid, mode)

def makeGaussianPyramidHelper(im, scale, minsize, pyramid, mode) :
  height, width = im.size
  im_array = np.asarray(im, dtype = np.float64)
  pyramid.append(im_array)
  if (height * scale <= minsize and width * scale <= minsize):
    return pyramid
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
  return makeGaussianPyramidHelper(nextImage, scale, minsize, pyramid, mode)

def showGaussianPyramid(pyramid):
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

def resizeImage(template, width):
  orig_height, orig_width = template.size
  scale = width/orig_width
  height = orig_height * scale
  it_array = np.asarray(template, dtype = np.float64)
  it_array2 = it_array.copy()
  output_array = it_array.copy()
  output_array = gaussian_filter(it_array2, 1/(2*scale))
  filtered_it = Image.fromarray(output_array.astype('uint8'))
  new_template = filtered_it.resize((int(height), int(width)), Image.BICUBIC)
  return new_template

def drawAt(image, col, row, height, width):
  if image.mode is "L":
    image.convert("RGB")
    # print(image.mode)
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



def detectFace(pyramid, template, threshold):
  orig_im_array = pyramid[0]
  orig_im_array2 = orig_im_array.copy()
  face_locations = []
  # adjust template size
  template_im = Image.open(template)
  new_template = resizeImage(template_im, TEMPLATE_SIZE)
  orig_im = Image.fromarray(pyramid[0].astype('uint8'))
  orig_im2 = orig_im.copy()
  template2 = new_template.copy()
  t_width, t_height = template2.size
  orig_width, orig_height = orig_im.size
  for image_array in pyramid:
    # load image from pyramid
    im = Image.fromarray(image_array.astype('uint8'))
    # calculate ncc array of the image
    ncc_array = normxcorr2D(im, template2)
    im_width, im_height = im.size
    scale = orig_height / im_height
    nccBin_array = ncc_array > threshold
    # draw at where ncc value > threshold
    for row in range(0, ncc_array.shape[0]):
      for col in range(0, ncc_array.shape[1]):
        if ncc_array[row, col] > threshold:
          newRow = int(row * scale)
          newCol = int(col * scale)
          face_locations.append((newRow, newCol, scale))
          drawAt(orig_im2, newCol, newRow,
                 t_height * scale, t_width * scale)
  image = Image.fromarray(orig_im_array2, "RGB")
  orig_im2.save("faceDetection.png", "PNG")
  # return orig_im2
  return face_locations

def swapFace(orig_image, face_locations, face):
  orig_im_array = np.asarray(orig_image, dtype = 'uint8')
  output_im_array = orig_im_array.copy()
  face_array = np.asarray(face, dtype = 'uint8')
  for face_loc in face_locations:
    output_im_array = swapFaceHelper(face_loc, output_im_array, face_array)
  output_im = Image.fromarray(output_im_array.astype('uint8'))
  return output_im

def swapFace(orig_image, face_locations, face, index):
  im = Image.open(orig_image)
  orig_im_array = np.asarray(im, dtype = 'uint8')
  output_im_array = orig_im_array.copy()
  face_im = Image.open(face)
  # face_im = resizeImage(face_im, TEMPLATE_SIZE)
  # face_array = np.asarray(face_im, dtype = 'uint8')
  output_im_array = swapFaceHelper(face_locations[index], output_im_array, face_im)
  output_im = Image.fromarray(output_im_array.astype('uint8'))
  output_im.save("swappedImage.png", "PNG")
  return output_im


def swapFaceHelper(face_loc, output_im_array, face_im):
  row = face_loc[0]
  col = face_loc[1]
  scale = face_loc[2]
  print(f"this is scale: {scale}")
  width, height = face_im.size
  print(f"this is width&height: {face_im.size}")
  width_to_reach = TEMPLATE_SIZE * scale
  newscale = width_to_reach/width
  height_to_reach = height * newscale
  print(f"this is height to reach: {height_to_reach}")

  face_array = np.asarray(face_im, dtype = np.float64)
  face_array2 = face_array.copy()
  output_array = face_array.copy()
  output_array = gaussian_filter(face_array2, 1/(2*scale))
  filtered_face = Image.fromarray(output_array.astype('uint8'))
  new_face_im = filtered_face.resize((int(width_to_reach), int(height_to_reach)), Image.BICUBIC)

  face_array = np.asarray(new_face_im, dtype = 'uint8')
  print(f"this is face height: {height}")
  radiusW = math.floor(width_to_reach/2)
  radiusH = math.floor(height_to_reach/2)
  for x in range(0, int(height_to_reach)):
    for y in range(0, int(width_to_reach)):
      output_im_array[x + row - radiusH, y + col - radiusW] = face_array[x, y]
  return output_im_array

def faceSwap(image, template, face):
  if SetUpImages.setUpImages(image, template, face):
    pyramid = makeGaussianPyramid('b&wImage.png', 0.75, 50)
    face_locs = detectFace(pyramid, 'b&wTemplate.png', 0.9)
    swapFace('b&wImage.png', face_locs, 'b&wFace.png', 0)
    SetUpImages.smoothImage('swappedImage.png', 2)

faceSwap('girl.jpg', 'template_girl.png', 'face_to_change.png')