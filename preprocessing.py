import sqlite3

import numpy as np
from PIL import Image
import cv2
from tkinter import *
import sqlite3
import os
import sys
from PIL import ImageTk, Image
root = Tk()
root.geometry('1366x768')
root.title("Satellite")
root.configure(bg='white')
canv = Canvas(root, width=1366, height=768, bg='white')
canv.grid(row=2, column=3)
img = Image.open('back.png')
photo = ImageTk.PhotoImage(img)
canv.create_image(1,1, anchor=NW, image=photo)
from    matplotlib import pyplot as plt
# Reading color image as grayscale
conn = sqlite3.connect('Form.db')
with conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM imgsave")
    rows = cursor.fetchall()
    for row in rows:
        filename = row[0]



import os, sys
img1 = cv2.imread(filename)
gray = cv2.imread(filename,0)


img = cv2.imread(filename)

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)


img1 = Image.open(filename)
photo1 = ImageTk.PhotoImage(img1)
l1=Label(root, text="Preprocessing", bg="black",fg="white",font=("BOLD",20))
l1.place(x=700,y=200)
l1=Label(root, text="Original Image", bg="black",fg="white",font=("BOLD",12))
l1.place(x=700,y=250)

l1=Label(root, text="Gray Image", bg="black",fg="white",font=("BOLD",12))
l1.place(x=900,y=250)
b1=Button(root, text="Original Image",image=photo1,width=150,height=150)
b1.place(x=700,y=300)
image = Image.open(filename).convert("L")
photo2 = ImageTk.PhotoImage(image)
b1=Button(root, text="Original Image",image=photo2,width=150,height=150)
b1.place(x=900,y=300)
cv2.imwrite('dst.png', dst)
img1 = Image.open('dst.png')
photo3 = ImageTk.PhotoImage(img1)
l1=Label(root, text="Denoised Image", bg="black",fg="white",font=("BOLD",12))
l1.place(x=700,y=470)
b1=Button(root, text="Original Image",image=photo3,width=150,height=150)
b1.place(x=700,y=500)

image = cv2.imread(filename)
# create square shaped 7x7 pixel kernel
kernel = np.ones((7,7),np.uint8)

# dilate, erode and save results
dilated = cv2.dilate(image,kernel,iterations = 1)
eroded = cv2.erode(image,kernel,iterations = 1)
cv2.imwrite('morph-dilated.png', dilated)
cv2.imwrite('morph-eroded.png', eroded)

l1=Label(root, text="Morph-Dilated Image", bg="black",fg="white",font=("BOLD",12))
l1.place(x=900,y=470)
img1 = Image.open('morph-dilated.png')
photo4 = ImageTk.PhotoImage(img1)
b1=Button(root, text="Original Image",image=photo4,width=150,height=150)
b1.place(x=900,y=500)
l1=Label(root, text="Morph-Eroded", bg="black",fg="white",font=("BOLD",12))
l1.place(x=1100,y=470)

img1 = Image.open('morph-eroded.png')
photo5 = ImageTk.PhotoImage(img1)
b1=Button(root, text="Original Image",image=photo5,width=150,height=150)
b1.place(x=1100,y=500)



root.mainloop()