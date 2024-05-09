import numpy as np
import cv2 as cv,cv2
import  imutils
import sys
import pytesseract
import pandas as pd
import time

import subprocess

def run_ampy_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")
        
id_names = pd.read_csv('id-names.csv')
id_names = id_names[['id', 'name']]

faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')
NAME=""
lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)
lbph.read('Classifiers/TrainedLBPH.yml')
camera = cv.VideoCapture(0)
for _ in range(4):
    _, _ = camera.read()
while cv.waitKey(1) & 0xFF != ord('q'):
    _, img = camera.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        faceRegion = grey[y:y + h, x:x + w]
        faceRegion = cv.resize(faceRegion, (220, 220))

        label, trust = lbph.predict(faceRegion)
        try:
            name = id_names[id_names['id'] == label]['name'].item()
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(img, name, (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            NAME= name

        except:
            pass

    print(NAME)
    cv.imshow('Recognize', img)
    cv.waitKey(10000)
    if(NAME != ""):
        break
camera.release()
cv.destroyAllWindows()

###########################

print("Now I run the text detection code")
print("Turning the camera on in 5sec")
cv.waitKey(5000)
#################################

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.4.1/")
# Skip the first four frames
for _ in range(4):
    _, _ = cap.read()

# Capture the 5th frame
ret, img = cap.read()

# Check if the frame was successfully captured
if ret:
    # Save the image
    cv2.imwrite('image.jpeg', img)
    print("Image captured")
else:
    print("Error: Unable to capture the image.")
cap.release()
image = cv2.imread('image.jpeg')

image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gray = cv2.bilateralFilter(gray, 11, 17, 17)


edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None 

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)

# Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# Run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)

#Data is stored in CSV file
raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
        'v_number': [text]}

df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
df.to_csv('data.csv')

# Print recognized text
print(text)
print("passing control to ESP8266")
ampy_command = "ampy --port /dev/cu.usbserial-0001 --baud 115200 put main.py"

run_ampy_command(ampy_command)
print("PRESS RST on ESP8266")
cv2.waitKey(0)
cap.release()
cv.destroyAllWindows()



