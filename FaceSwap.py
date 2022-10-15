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
  if mode == "L":
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
  if mode == "L":
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
  if image.mode == "L":
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

def ncorr(image, template):
  t_array = np.asarray(template, dtype = 'float64')
  t_array2 = t_array.copy()
  im_array = np.asarray(image, dtype = 'float64')
  im_array2 = im_array.copy()
  ncorr = im_array.copy()

  for i in range(3):
    # modify only one channel of template and image
    tc_array = t_array2[:,:,i]
    imc_array = im_array2[:,:,i]
    tc_array = tc_array - np.mean(tc_array)
    norm = math.sqrt(np.sum(np.square(tc_array)))
    tc_array = tc_array/norm

    filter = np.ones(tc_array.shape)

    imcsq_array = np.square(imc_array)

    imcsum_array = signal.correlate2d(imc_array, filter, 'same')
    imcsqsum_array = signal.correlate2d(imcsq_array, filter, 'same')

    numer = signal.correlate2d(imc_array, tc_array, 'same')
    denom = np.sqrt(imcsqsum_array - np.square(imcsum_array)/np.size(tc_array))

    tol = np.sqrt(np.finfo(denom.dtype).eps)
    ncorr[:,:,i] = np.where(denom < tol, 0, numer/denom)
    ncorr2 = ncorr[:,:,i]
    ncorr[:,:,i] = np.where(np.abs(ncorr2 - 1.) > np.sqrt(np.finfo(ncorr2.dtype).eps), ncorr2, 0)

  return ncorr



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
    ncc_array = ncorr(im, template2)
    im_width, im_height = im.size
    scale = orig_height / im_height
    for i in range(3):
      ncc_array_channel = ncc_array[:,:,i]
      # draw at where ncc value > threshold
      for row in range(0, ncc_array_channel.shape[0]):
        for col in range(0, ncc_array_channel.shape[1]):
          if ncc_array_channel[row, col] > threshold:
            newRow = int(row * scale)
            newCol = int(col * scale)
            face_locations.append((newRow, newCol, scale))
            drawAt(orig_im2, newCol, newRow,
                  t_height * scale, t_width * scale)
  orig_im2.save("faceDetection.png", "PNG")
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
  output_im_array = swapFaceHelper(face_locations[index], output_im_array, face_im)
  output_im = Image.fromarray(output_im_array.astype('uint8'))
  output_im.save("swappedImage.png", "PNG")
  return output_im


def swapFaceHelper(face_loc, output_im_array, face_im):
  row = face_loc[0]
  col = face_loc[1]
  scale = face_loc[2]
  width, height = face_im.size
  width_to_reach = TEMPLATE_SIZE * scale
  newscale = width_to_reach/width
  height_to_reach = height * newscale

  face_array = np.asarray(face_im, dtype = 'float64')
  face_array2 = face_array.copy()
  output_array = face_array.copy()
  for i in range(3):
    output_array[:,:,i] = gaussian_filter(face_array2[:,:,i], 1/(2*scale))
  
  filtered_face = Image.fromarray(output_array.astype('uint8'))
  new_face_im = filtered_face.resize((int(width_to_reach), int(height_to_reach)), Image.BICUBIC)

  face_array = np.asarray(new_face_im, dtype = 'uint8')
  radiusW = math.floor(width_to_reach/2)
  radiusH = math.floor(height_to_reach/2)
  for i in range(3):
    for x in range(0, int(height_to_reach)):
      for y in range(0, int(width_to_reach)):
        output_im_array[x + row - radiusH, y + col - radiusW, i] = face_array[x, y, i]

  return output_im_array

def faceSwap(image, template, face, threshold):
  pyramid = makeGaussianPyramid(image, 0.75, 50)
  face_locs = detectFace(pyramid, template, threshold)
  swapFace(image, face_locs, face, 0)

faceSwap('girl.jpg', 'template_girl.png', 'face_to_change.png', 0.9)