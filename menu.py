from tkinter import *
import sqlite3
import os
from tkinter.filedialog import askopenfilename
import cv2
from PIL import ImageTk, Image
root = Tk()
root.geometry('1366x768')
root.title("Satellite")

canv = Canvas(root, width=1366, height=768, bg='white')
canv.grid(row=2, column=3)
img = Image.open('back.png')
photo = ImageTk.PhotoImage(img)
canv.create_image(1,1, anchor=NW, image=photo)
Un = StringVar()
Pw = StringVar()

def back():
    root.destroy()




def pre():

        os.system('python preprocessing.py')
def captimg():
    filename = askopenfilename(filetypes=[("images", "*.*")])
    img = cv2.imread(filename)
    cv2.imshow("Satellite", img)  # I used cv2 to show image
    cv2.waitKey(0)
    conn = sqlite3.connect('Form.db')
    cursor = conn.cursor()
    cursor.execute('delete from imgsave1')
    cursor.execute('INSERT INTO imgsave1(img ) VALUES(?)', (filename,))

    conn.commit()
    os.system("python div.py")
def clf():

    os.system('python classification.py')

def img():
    filename = askopenfilename(filetypes=[("images", "*.*")])
    print(filename)
    img = cv2.imread(filename)
    conn = sqlite3.connect('Form.db')
    cursor = conn.cursor()
    cursor.execute('delete from imgsave')
    cursor.execute('INSERT INTO imgsave(img ) VALUES(?)', (filename,))

    conn.commit()
    cv2.imshow("Satellite", img)  # I used cv2 to show image
    cv2.waitKey(0)
def seg():
    os.system('python seg.py')
def clf():
    os.system('python clf.py')

def featext():
    os.system('python featext.py')
def dispred():

    os.system('python dispred.py')
Button(root, text='Select Arial Image', width=30,height=2, bg='green', fg='white', command=captimg, font=("bold", 10)).place(x=800, y=350)
Button(root, text='Select Area', width=30,height=2, bg='green', fg='white', command=img, font=("bold", 10)).place(x=800, y=400)
Button(root, text='Preprocessing', width=30,height=2, bg='green', fg='white', command=pre, font=("bold", 10)).place(x=800, y=450)
Button(root, text='Segmentation', width=30,height=2, bg='green', fg='white', command=seg, font=("bold", 10)).place(x=800, y=500)
Button(root, text='Feature Extraction', width=30,height=2, bg='green', fg='white', command=featext, font=("bold", 10)).place(x=800, y=550)
Button(root, text='Classification', width=30,height=2, bg='green', fg='white', command=clf, font=("bold", 10)).place(x=800, y=600)

root.mainloop()
