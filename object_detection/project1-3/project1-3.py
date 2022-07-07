#selective search을 통한 Object Detection
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import numpy as np 

img = cv2.imread('dog_image.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape:', img.shape)


# Region Proposal 
_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=1000)
cand_rects = [cand['rect'] for cand in regions]
'''  
img_rgb_copy = img_rgb.copy()

for rect in cand_rects:
    
    left = rect[0]
    top = rect[1]
    right = left + rect[2]
    bottom = top + rect[3]
    
    img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color=(125, 255, 51), thickness=2)

plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_copy)
plt.show(block=False)
plt.pause(1)
plt.close()
'''  

#정답 데이터 설정
img_rgb_copy2 = img_rgb.copy()

correct_left = 70
correct_top =  350
correct_right = 470
correct_bottom = 600
cv2.rectangle(img_rgb_copy2, (correct_left, correct_top), (correct_right, correct_bottom), color=(0, 0, 255), thickness=2)


for rect in cand_rects:

    left = rect[0]
    top = rect[1]
    right = left + rect[2]
    bottom = top + rect[3]
    # Calculate IoU
    x1 = np.maximum(left, correct_left)
    y1 = np.maximum(top, correct_top)
    x2 = np.minimum(right, correct_right)
    y2 = np.minimum(bottom, correct_bottom)
    
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    cand_box_area = rect[2] * rect[3]
    gt_box_area = (correct_right - correct_left) * (correct_bottom - correct_top)
    union = abs(cand_box_area) + abs(gt_box_area)- abs(intersection)
    print(intersection)
    IoU = intersection / union
    if IoU > 0.5:
        cv2.rectangle(img_rgb_copy2, (left, top), (right, bottom), color=(125, 255, 51), thickness=2)
        text = str(IoU)[0:5]
        cv2.putText(img_rgb_copy2,text, (left+30, top+20), cv2.FONT_HERSHEY_SIMPLEX,0.6, color=(125, 255, 51), thickness=2)

img_rgb = cv2.cvtColor(img_rgb_copy2, cv2.COLOR_BGR2RGB)
cv2.imwrite('result.png',img_rgb)
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_copy2)
plt.show()

