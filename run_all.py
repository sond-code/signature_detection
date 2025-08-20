from pdf2image import convert_from_path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from run_extraction import run_extraction
from tesseract import check_context


image_count  = 0  
def view_image(image,title,mask = False):
    if not mask:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image,cmap = 'gray')
    plt.axis('off')
    plt.title(f"page  {title}") 
    plt.show()

def add_label(img,text,x,y):
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    color = (255, 0 ,0)  # Green color (BGR)
    thickness = 2
    lineType = cv2.LINE_AA
    org = (x,y)
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)

classes_arr = ['human_text' , 'signature']

signatures_arr=[] 
text_arr = defaultdict(list)
def run_inference(img):
    global signatures_arr
    global text_arr
    global image_count  
    image_count+=1
    print(" current image" , image_count)
    results = model(img)[0]
    for box in results.boxes:
        app_flg = False
        box_class  = box.cls[0].cpu().numpy().astype(int)
        print("box class " , box_class)
        if box_class:
            app_flg = True
        box_conf = box.conf[0] 
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        print (f" class {box_class}  --- confidence {box_conf} --- xyxy {xyxy} ")
        color = (0,255,0)
        if xyxy is not None and len(xyxy) >=4 :
            top_left  = (int(xyxy[0]),int(xyxy[1]))
            bottom_right  = (int(xyxy[2]),int(xyxy[3]))
            if box_class == 1:
                color = (0,0,255)
           
            cv2.rectangle(img,top_left, bottom_right, color,  2)
            label = classes_arr[box_class]
            add_label(img,label, top_left[0] , top_left[1])
            if app_flg:
                signatures_arr.append([top_left,bottom_right,image_count])
            else:
                text_arr[image_count].append((top_left,bottom_right))
            #cropped_with_sig = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            #view_image(cropped_with_sig,'cropped image')
            ##### will need the  page number in the image too as well as the signature bbox to crop #####

img_path = 'test_2.pdf'
many  = False
if img_path.endswith(".pdf"):
    many = True
    print("pdf doc")
    images = convert_from_path(img_path)
else :
    print("one image")
    images = cv2.imread(img_path)
    print("images length " , len(images))
model = YOLO('human_sig_weight_final.pt')
if many :
    print( "lengrth more than one")
    for idx,img in enumerate(images):
        image = np.array(img)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        run_inference(image)
        view_image(image,idx)  
    print("checking signatures array")
    if signatures_arr:
        print("going thought the signatures array now")
        for elem in signatures_arr:
            top_left = elem[0]
            bottom_right = elem[1]
            idx = elem[2] - 1 
            image_to_crop = np.array(images[idx])
            cropped_sig = image_to_crop[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
            print("feeding to segmentation model")
            view_image(cropped_sig,"signature",True)
            mask = run_extraction(cropped_sig)
            view_image(mask,'mask',True)

            ### feed to the segmentation model to get corresponsding mask -----   ### 

    else:
        print("cant find signatures --- defaulting to lLM instead! ")
        print('*'*10)
        res = check_context(text_arr,img_path)
        for signature in res:
            signature_page = signature[0]
            signature_idx = signature[1]
            top_left , bottom_right = text_arr[signature_page][signature_idx]
            final = np.array(images[signature_page-1])
            im = np.array(images[signature_page-1])
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            cv2.rectangle(im, top_left, bottom_right,(0,255,0) , 2)    
            view_image(im,"llm based") 
else:
    print("single imge")
    run_inference(images)
    view_image(images,0)



