import os
import cv2
import numpy as np

from tkinter import Tk
from tkinter.filedialog import askopenfilename

curr_iter = 0
crop_folder = 'cropped_imgs'
os.makedirs(crop_folder, exist_ok=True)
image_map = {}

def get_img_name(img):
    splitted = img.split('/')
    return splitted[len(splitted)-1].split('.')[0]

def process_coordinates(coords):
    try: 
        min_x =  max_x = coords[0][0]
        min_y =  max_y = coords[0][1]
    except Exception as error:
        raise error
    for coord in coords:
        x_cord = coord[0]
        y_cord = coord[1]
        min_x = min(x_cord,min_x)
        max_x = max(x_cord,max_x)
        min_y = min(y_cord,min_y)
        max_y = max(y_cord,max_y)
    return [(min_x,min_y) , (max_x,max_y)]

while(1):
    cv2.destroyAllWindows()

    if curr_iter == 0:
        Tk().withdraw()
        img_path = askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.webp")])
    else:
        u_input = input(" press P for using previous --- press N for reselecting ")
        if u_input.lower() == 'n': 
            Tk().withdraw()
            img_path = askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        else:
            print("using prevous image >> " ,img_path)

    image_map[img_path] = image_map.get(img_path,0) + 1
    print('image map now' , image_map)
    current_clicks = 0
    coords = []
    img = cv2.imread(img_path)
    img_display = img.copy()

    def mouse_click(event, x, y, flags, param):
        global current_clicks
        global coords
        if event == cv2.EVENT_LBUTTONDOWN:
            current_clicks +=1
            print("click count >>  " , current_clicks)
            coords.append((x,y))
            print("appended coordinates >> " , x,y)
            cv2.circle(img_display, (x, y), 1, (0, 0, 255), 20)

    cv2.namedWindow("crop_sig", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("crop_sig", 1000, 800)
    cv2.setMouseCallback('crop_sig', mouse_click)

    while(1) :
        cv2.imshow('crop_sig',img_display)
        key = cv2.waitKey(1)
        if current_clicks == 4 or  key == 27:
            print("hit threshold 4 >> drawing box now")
            break

    visited_images = set()
    print ("current coords " , coords)
    final = process_coordinates(coords)
    print  ("final coordinates >>> " , final)
    cropped_img = img[final[0][1]:final[1][1], final[0][0]:final[1][0]]

    while(1):
        cv2.imshow('cropped_final',cropped_img)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyWindow('crop_sig')
            break

    user_input = input('Y to confirm -- N to ignore: ')
    if user_input.lower() == 'y':
        print("Confirmed. -- saving image ")
        print("inage path " , img_path)
        img_name = get_img_name(img_path)
        img_final_path = img_name + '_cropped' + str(image_map[img_path]) +'.jpg'
        save_path = os.path.join(crop_folder,img_final_path)
        cv2.imwrite(save_path, cropped_img)
    curr_iter+=1
