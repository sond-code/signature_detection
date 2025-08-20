import cv2

# Load image in grayscale (single channel)
img = cv2.imread('masks/img_mask.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold to binary (if needed, depending on your mask)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours
thresh_inv = cv2.bitwise_not(thresh)
contours, hierarchy = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Create a blank image to draw contours on (same size as original)
contour_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # convert to BGR for color drawing

# Draw contours (in green, thickness 2)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Display loop
while True:
    cv2.imshow("Original Image", img)
    cv2.imshow("Contours", contour_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to quit
        break

cv2.destroyAllWindows()
