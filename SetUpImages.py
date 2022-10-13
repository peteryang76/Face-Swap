from PIL import Image
import cv2
import threading

def setUpImages(image, template, face):
    event = threading.Event()
    im = Image.open(image)
    im.convert("L")
    event.wait(20)
    im.save("b&wImage.png", "PNG")
    temp_im = Image.open(template)
    temp_im.convert("L")
    event.wait(10)
    temp_im.save("b&wTemplate.png", "PNG")
    face_im = Image.open(face)
    face_im.convert("L")
    event.wait(10)
    face_im.save("b&wface.png", "PNG")
    return True

def smoothImage(image, times):
    event = threading.Event()
    event.wait(10)
    im = cv2.imread(image)
    for i in range(times):
        im = cv2.medianBlur(im, 7)
    output_im = Image.fromarray(im)
    output_im.save("swappedImage.png", "PNG")
    return output_im
