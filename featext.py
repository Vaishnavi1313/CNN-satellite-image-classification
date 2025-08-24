import sqlite3

import cv2
conn = sqlite3.connect('Form.db')
with conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM imgsave")
    rows = cursor.fetchall()
    for row in rows:
        filename = row[0]
img = cv2.imread(filename, 0)
img = cv2.resize(img, (450, 300))


def null(x):
    pass


# create trackbars to control threshold values
cv2.namedWindow('Canny')
cv2.resizeWindow('Canny', (450, 300))
cv2.createTrackbar('MIN', 'Canny', 80, 255, null)
cv2.createTrackbar('MAX', 'Canny', 120, 255, null)
while True:
    # get Trackbar position
    a = cv2.getTrackbarPos('MIN', 'Canny')
    b = cv2.getTrackbarPos('MAX', 'Canny')
    # Canny Edge detection
    # arguments: image, min_val, max_val
    canny = cv2.Canny(img, a, b)
    # display the images
    cv2.imshow('Canny', canny)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cv2.destroyAllWindows()
