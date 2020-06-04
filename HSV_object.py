import cv2 as cv
import numpy as np

def pos(x):
    pass

cv.namedWindow('tracking')

cv.createTrackbar('l_h', 'tracking', 0, 255, pos)
cv.createTrackbar('l_s', 'tracking', 0, 255, pos)
cv.createTrackbar('l_v', 'tracking', 0, 255, pos)
cv.createTrackbar('u_h', 'tracking', 0, 255, pos)
cv.createTrackbar('u_s', 'tracking', 0, 255, pos)
cv.createTrackbar('u_v', 'tracking', 0, 255, pos)

cap = cv.VideoCapture(0)

faces_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


while 1:
    ret, frame = cap.read()
    # frame = cv.resize(frame, (1024, 576))
    frame1 =frame
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    hsv = cv.medianBlur(hsv, 31)
    hsv = cv.bilateralFilter(hsv, 9, 75, 75)

    l_h = cv.getTrackbarPos('l_h', 'tracking')
    l_s = cv.getTrackbarPos('l_s', 'tracking')
    l_v = cv.getTrackbarPos('l_v', 'tracking')
    u_h = cv.getTrackbarPos('u_h', 'tracking')
    u_s = cv.getTrackbarPos('u_s', 'tracking')
    u_v = cv.getTrackbarPos('u_v', 'tracking')

    l_b = np.array([0, 110, 144])
    u_b = np.array([255, 255, 255])
    # l_b = np.array([l_h, l_s, l_v])
    # u_b = np.array([u_h, u_s, u_v])

    mask = cv.inRange(hsv, l_b, u_b)

    res = cv.bitwise_and(frame, frame, mask=mask)


    cv.imshow('mask', mask)
    cv.imshow('res', res)

    kernel = np.ones((4, 4), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    canny = cv.Canny(opening, 60, 60*3)
    cv.imshow('canny', canny)

    con, hier = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in con:
        (x, y), radius = cv.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        frame = cv.circle(frame, center, radius, (255, 0, 0), 2)

        font = cv.FONT_HERSHEY_SIMPLEX
        frame = cv.putText(frame, 'Orange', (int(x), int(y+50)), font, .5, (0, 255, 0), 2)
        pos = str(int(x))+' '+str(int(y))+'  '
        frame = cv.putText(frame, pos, (0, 20), font, 0.7, (0, 0, 255), 1)

        if int(x) >= 0 and int(x) <= 213:
            frame = cv.putText(frame, '         Left', (0, 25), font, .7, (0, 0, 255), 2)

        if int(x) >= 214 and int(x) <= 427:
            frame = cv.putText(frame, '                          Middle', (0, 25), font, .7, (0, 0, 255), 2)

        if int(x) >= 428 and int(x) <= 640:
            frame = cv.putText(frame, '                                         Right', (0, 25), font, .7, (0, 0, 255), 2)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faces_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        frame = cv.putText(frame, 'face', (int(x), int(y)), font, 1, (0, 255, 0), 2)

    cv.imshow('frame', frame)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()