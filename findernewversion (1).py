import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('fotki/many.avi')
sila_pixes = []
index_sila = np.arange(256)
x = []
y = []
coords = []
kon = 0
while True:
    try:
        ret, frame = cap.read()
        frame = np.array(frame)
        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        mask = np.zeros(frame.shape, np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresholding_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresholding_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour, False)
            sum_of_sila = M['m00']
            # M['m00'] = 0
            # M["m10"] = 0
            # M["m01"] = 0
            for j in contour[0]:
                # M['m00'] = M['m00'] +  gray[j[0], j[1]]
                # M['m10'] = M['m10'] + j[0]
                if M['m00'] != 0:
                    xcenter = int(M["m10"] / M['m00'])
                    ycenter = int(M["m01"] / M['m00'])
                    cv2.circle(frame, (xcenter, ycenter), 3, (0, 0, 255), -1)

        cv2.imshow('frame', frame)
        cv2.imshow('binary', thresholding_img)

        currentname = f'frame{kon}'
        cv2.imwrite(f'C:/Users/probnik/Desktop/opencvvvv/kadry/frame{currentname}.png', frame)  # Пропишите здесь свой путь как папке с кадрами
        kon += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    except IndexError:
        print('Index out of range')

