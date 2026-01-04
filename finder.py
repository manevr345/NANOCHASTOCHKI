import cv2
import numpy as np

cv2.namedWindow('Result: Original')
cap = cv2.VideoCapture('test.avi')

kon =0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

    frame = frame[200:450, 400:800 ]
    frame = cv2.resize(frame, (frame.shape[1] *2, frame.shape[0] * 2))
    contur = np.zeros(frame.shape, np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.Canny(gray, 25, 255)

    # В findContours я думаю можно будет поиграться с апроксимацией

    con, hir = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, con, -1, (45, 23, 255), 1, cv2.LINE_AA)
    cv2.imshow('Result: Original', frame)
    cv2.imshow('Result: Original', frame)

    # Покадровое разбиение я добавил для проверки качества программы

    currentname = f'frame{kon}'
    cv2.imwrite(f'C:/Users/probnik/Desktop/opencvvvv/kadry/frame{currentname}.png', frame) #  Пропишите здесь свой путь как папке с кадрами
    kon += 1

    cv2.imshow('Result: Original', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
