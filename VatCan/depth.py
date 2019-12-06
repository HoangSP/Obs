import cv2
import numpy as np

def biggestContourI(contours):
    maxVal = 0
    maxI = None
    for i in range(0, len(contours) - 1):
        if len(contours[i]) > maxVal:
            cs = contours[i]
            maxVal = len(contours[i])                                                                                                                                                                                                                                                                                                                                                                                                                                   
            maxI = i
    return maxI

def find_obstacle(img):
    k = []
    print(img.shape)
    img.shape[:2]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            if k < 128 or k > 146:
                img[i][j] = 0
    return img

def remove_groud(img):
    h, w = img.shape[:2]
    newImg = img.copy()
    for i in range(h-1,0,-1):
        for j in range(w-1, 0,-1):
            if img[i, j] - img[i-7, j] > 5  and img[i,j] >0 or img[i,j] > 150:
                newImg[i,j ] = 0
    return newImg
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgContour):
    _,cnts,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        area = cv2.contourArea(cnt)
        print(area)
        print(len(cnt))
        if len(cnt) >= 4 and (area > 400 or area < 2800):
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2)  
def roi(img):
    h,w = img.shape[:2]
    crop = img[h//3:w*2//3,:]
    return crop
# def pr(img)
    
for i in range(57):
    img = cv2.imread('./obstacle/obstacle_{}.jpg'.format(i))
    crop = roi(img)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blurred,134,255)
    ret,th = cv2.threshold(blurred,134,255, cv2.THRESH_BINARY_INV)
    
    rmg = remove_groud(blurred)
    kernelOpen = np.ones((30,30),np.uint8)
    kernelClose = np.ones((30,30),np.uint8)
    opening = cv2.morphologyEx(rmg, cv2.MORPH_OPEN, kernelOpen)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernelClose)
    imgContour = crop.copy()
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(opening, kernel, iterations=1)
    getContours(imgDil,imgContour)
    imgStack = stackImages(0.8,([rmg,closing],[opening,imgContour]))
    cv2.imshow("Result", imgStack)
    # cv2.imshow("gray", gray)
    # cv2.imshow("remove_groud", rmg)
    # cv2.imshow("opening", opening)
    key = cv2.waitKey(1)
    if key == 27:
        break


