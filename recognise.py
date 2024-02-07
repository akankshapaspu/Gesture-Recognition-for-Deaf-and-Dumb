import cv2
import numpy as np

def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'C'
       elif result[0][3] == 1:
              return 'D'
       elif result[0][4] == 1:
              return 'E'
       elif result[0][5] == 1:
              return 'F'
       elif result[0][6] == 1:
              return 'G'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'K'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][12] == 1:
              return 'M'
       elif result[0][13] == 1:
              return 'N'
       elif result[0][14] == 1:
              return 'O'
       elif result[0][15] == 1:
              return 'P'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'R'
       elif result[0][18] == 1:
              return 'S'
       elif result[0][19] == 1:
              return 'T'
       elif result[0][20] == 1:
              return 'U'
       elif result[0][21] == 1:
              return 'V'
       elif result[0][22] == 1:
              return 'W'
       elif result[0][23] == 1:
              return 'X'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'
     


cam = cv2.VideoCapture(0)

text=''

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    cv2.putText(frame,text,(100, 400),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0, 0, 255),5)
    img = cv2.rectangle(frame, (425,100),(625,300), (0,0,255), thickness=3, lineType=8, shift=0)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Detect Hand Gestures", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "1.png"
        imcrop = img[103:297, 428:622]
        bw_img=cv2.cvtColor(imcrop.copy(),cv2.COLOR_BGR2GRAY)
        gaussian=cv2.GaussianBlur(bw_img,(11,11),0)
        thresholded=cv2.adaptiveThreshold(gaussian,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        imgCopy = thresholded.copy()
        contours, hierarchy=cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        #sort the contours as per the area
        contours=sorted(contours,key=cv2.contourArea,reverse=True)
        contours=[contours[0]]
        
        mask=np.zeros(imcrop.shape,dtype='uint8')
        mask=cv2.drawContours(mask,contours,-1,(255,255,255),thickness=cv2.FILLED)
        mask=cv2.bitwise_not(mask)
        mask=cv2.resize(mask,(image_x,image_y))
        cv2.imshow("Mask",mask)
        cv2.imwrite(img_name,mask)
        
        predictedAlpha=predictor()
        if predictedAlpha!=None:
            text+=predictor()
        # cv2.putText(frame,text,(100, 400),cv2.FONT_HERSHEY_TRIPLEX,1,(255, 0, 0),20,cv2.LINE_AA)
        print(text)
        
    elif k%256 == 100:
        text=text[0:len(text)-1]
        
    elif k%256 == 115:
        text+=' '
        
        

cam.release()

cv2.destroyAllWindows()