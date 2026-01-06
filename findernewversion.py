import cv2
import numpy as np

cap = cv2.VideoCapture('test.avi')
sila_pixes = []
index_sila = np.arange(256)
center = []
kon = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    frame = frame[200:450, 400:800 ]
    frame = cv2.resize(frame, (frame.shape[1] *2, frame.shape[0] * 2))
    mask = np.zeros(frame.shape, np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresholding_img = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    coords = np.where( thresholding_img == 255 )
    coords =list(zip(coords[0], coords[1]))

    for i in coords:
        for j in index_sila:
            if gray[*i] == j:
                sila_pixes.append(j)
    result = list(zip(coords, sila_pixes))

    proxix = [data[1] * data[0][0] for data in result]
    proxiy = [data[1] * data[0][1] for data in result]

    sum_sila = sum([data[1] for data in result])
    if sum_sila != 0:
        x_center = sum(proxix) / sum_sila
        y_center = sum(proxiy) / sum_sila
    else:
        x_center = 0
        y_center = 0
    cv2.circle(frame, ( int(y_center), int(x_center)), 2, (255, 0, 0), 1)


    cv2.imshow('frame', frame)

    currentname = f'frame{kon}'
    cv2.imwrite(f'C:/Users/probnik/Desktop/opencvvvv/kadry/frame{currentname}.png',
                frame)  # Пропишите здесь свой путь как папке с кадрами
    kon += 1

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
