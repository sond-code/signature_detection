import cv2
import numpy as np
import os
from scipy.interpolate import splprep, splev

def extract_name(image_name):
    if '/' in image_name:
        return  image_name.split('/')[1].split('.')
    return image_name.split('.')
number = 1
image_name = 'cropped_imgs/large (7)_cropped9.jpg'
img_prefix = extract_name(image_name)[0]
img_postfix = extract_name(image_name)[1]
save_image_path = img_prefix + "_" + str(number) +  "." + img_postfix
save_mask_path = img_prefix +  "_" +  str(number) + "_mask" +  "." + img_postfix
mask_folder = 'masks'
images_folder = 'images'
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)
component_points = []



# Load image
img = cv2.imread(image_name)
if img is None:
    raise FileNotFoundError(f"Image '{image_name}' not found.")

h, w, _ = img.shape
print("Image shape:", h, w)

# Create white mask
image_mask = np.ones((h, w), dtype=np.uint8) * 255

cv2.namedWindow("create_sig")
drawing_active = False

# Realistic pen thickness (2 pixels = normal pen)
PEN_THICKNESS = 1

def smooth_points_spline(points, num_points=500):
    """Smooth points using B-spline interpolation"""
    if len(points) < 4:
        return points
    
    points = np.array(points)
    
    try:
        tck, u = splprep([points[:, 0], points[:, 1]], s=1.0, per=False, k=min(3, len(points)-1))
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        smooth_points = np.column_stack([x_new, y_new]).astype(np.int32)
        return smooth_points
    except:
        return points

def connect(points_arr):
    """Connect points with smooth curves"""
    if len(points_arr) < 2:
        return
    
    # Get smooth points
    smooth_points = smooth_points_spline(points_arr)
    
    # Draw on both image and mask with same thickness
    cv2.polylines(img, [smooth_points], False, (0, 0, 255), PEN_THICKNESS, cv2.LINE_AA)
    cv2.polylines(image_mask, [smooth_points], False, 0, PEN_THICKNESS, cv2.LINE_AA)

# Mouse drawing
last_point = None

def mouse_callback(event, x, y, flags, param):
    global drawing_active, component_points, last_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_active = True
        component_points = [(x, y)]
        last_point = (x, y)
        
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing_active:
        if last_point:
            dist = np.sqrt((x - last_point[0])**2 + (y - last_point[1])**2)
            if dist > 2:
                component_points.append((x, y))
                cv2.line(img, last_point, (x, y), (0, 0, 255), PEN_THICKNESS, cv2.LINE_AA)
                last_point = (x, y)
                
    elif event == cv2.EVENT_LBUTTONUP:
        if len(component_points) > 1:
            # Just draw the smooth curve
            connect(component_points)
        
        component_points = []
        drawing_active = False
        last_point = None

cv2.setMouseCallback('create_sig', mouse_callback)

print("\nControls:")
print("- Draw with mouse")
print("- Press 'm' to view mask")
print("- Press 'o' to view original")
print("- Press 'c' to clear")
print("- Press ESC to save and exit")

show_mask = False

thick_arr = []

for i in range(1,4):
    thick_arr.append(ord(str(i)))

while True:
    if show_mask:
        cv2.imshow('create_sig', image_mask)
    else:
        cv2.imshow('create_sig', img)
    
    key = cv2.waitKey(1)
    
    if key == 27:  # ESC
        break

    elif key in thick_arr:
        PEN_THICKNESS = key - ord('0')
        print ('current thickness :', PEN_THICKNESS)

    elif key == ord('m'):
        show_mask = True
    elif key == ord('o'):
        show_mask = False
    elif key == ord('c'):
        img = cv2.imread(image_name)
        image_mask = np.ones((h, w), dtype=np.uint8) * 255
        show_mask = False

cv2.destroyAllWindows()

user_input = input('Save the signature mask? (y/n): ').strip().lower()

if user_input == 'y':
    name_parts = extract_name(image_name)
    mask_filename = f"{name_parts[0]}_mask.{name_parts[1]}"
    save_mask_path = os.path.join(mask_folder, save_mask_path)
    save_image_path = os.path.join(images_folder, save_image_path)
    
    try:
        cv2.imwrite(save_mask_path, image_mask)
        cv2.imwrite(save_image_path, img)
        print(f"Mask saved to {save_mask_path}")
        print(f"image saved to {save_image_path}")

    except Exception as e:
        print(f"Error saving: {e}")
else:
    print("Not saved.")