import sqlite3

import cv2, time
conn = sqlite3.connect('Form.db')
with conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM imgsave1")
    rows = cursor.fetchall()
    for row in rows:
        filename = row[0]
img = cv2.imread(filename)
img2 = img

height, width, channels = img.shape
# Number of pieces Horizontally
W_SIZE = 3
# Number of pieces Vertically to each Horizontal
H_SIZE = 3

for ih in range(H_SIZE):
    for iw in range(W_SIZE):
        x = width / W_SIZE * iw
        y = height / H_SIZE * ih
        h = (height / H_SIZE)
        w = (width / W_SIZE)
        print(x, y, h, w)
        img = img[int(y):int(y + h), int(x):int(x + w)]
        NAME = str(time.time())
        cv2.imwrite("sat1/" + str(ih) + str(iw) + ".png", img)
        img = img2