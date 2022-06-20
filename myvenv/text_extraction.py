import matplotlib.pyplot as plt
import IPython.display as Disp
from ipywidgets import widgets
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
#from PIL import *
#import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#C:\Program Files\Tesseract-OCR     path of tesseract installation

class bbox_select():
   # %matplotlib notebook 


    def __init__(self,im):
        self.im = im
        self.selected_points = []
        self.fig,ax = plt.subplots()
        self.img = ax.imshow(self.im.copy())
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        Disp.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)


        
    def poly_img(self,img,pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)),7)
        return img

    def onclick(self, event):
    #display(str(event))
        self.selected_points.append([event.xdata,event.ydata])
        if len(self.selected_points)>1:
            self.fig
            self.img.set_data(self.poly_img(self.im.copy(),self.selected_points))
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)

im = plt.imread('test1.png')

bs = bbox_select(im)

bs.selected_points = [[1562.2667728768347, 1475.0618435546166],
  [1575.3845691798433, 1619.3576028877083],
  [954.4755441707819, 1614.9850041200389],
  [971.9659392414596, 1479.434442322286]]

arr = np.array([bs.selected_points],'int')
print(arr)
mask = cv2.fillPoly(np.zeros(im.shape,np.uint8),arr,[1,1,1])
op = np.multiply(im,mask)
plt.imshow(op)


#Code to extract info
img = cv2.imread("test1.png")
crop_img = img[1000:1950, 900:1900] 
cv2.imshow("cropped", crop_img)
extractedInformation = pytesseract.image_to_string(Image.fromarray(crop_img))
print(extractedInformation)
cv2.waitKey(0)

# image = cv2.imread('test1.png')

# x,y,w,h = cv2.selectROI(image)

# print(x,y,w,h)

cv2.destroyAllWindows()



