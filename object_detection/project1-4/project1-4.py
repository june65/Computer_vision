# OpenCV Tensorflow Faster-RCNN

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('dog_image.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('image shape:', img.shape)
model_path = './pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'
config_path = './pretrained/config_graph.pbtxt'

cv_net = cv2.dnn.readNetFromTensorflow(model_path,config_path)

labels_to_names_0 = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
                    10:'fire hydrant',11:'street sign',12:'stop sign',13:'parking meter',14:'bench',15:'bird',16:'cat',17:'dog',18:'horse',19:'sheep',
                    20:'cow',21:'elephant',22:'bear',23:'zebra',24:'giraffe',25:'hat',26:'backpack',27:'umbrella',28:'shoe',29:'eye glasses',
                    30:'handbag',31:'tie',32:'suitcase',33:'frisbee',34:'skis',35:'snowboard',36:'sports ball',37:'kite',38:'baseball bat',39:'baseball glove',
                    40:'skateboard',41:'surfboard',42:'tennis racket',43:'bottle',44:'plate',45:'wine glass',46:'cup',47:'fork',48:'knife',49:'spoon',
                    50:'bowl',51:'banana',52:'apple',53:'sandwich',54:'orange',55:'broccoli',56:'carrot',57:'hot dog',58:'pizza',59:'donut',
                    60:'cake',61:'chair',62:'couch',63:'potted plant',64:'bed',65:'mirror',66:'dining table',67:'window',68:'desk',69:'toilet',
                    70:'door',71:'tv',72:'laptop',73:'mouse',74:'remote',75:'keyboard',76:'cell phone',77:'microwave',78:'oven',79:'toaster',
                    80:'sink',81:'refrigerator',82:'blender',83:'book',84:'clock',85:'vase',86:'scissors',87:'teddy bear',88:'hair drier',89:'toothbrush',
                    90:'hair brush'}

rows = img.shape[0]
cols = img.shape[1]
draw_img = img.copy()

cv_net.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))

cv_out = cv_net.forward()

for detection in cv_out[0,0,:,:]:
    score = float(detection[2])
    class_id = int(detection[1])
    if score > 0.7:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        caption = "{}: {:.3f}".format(labels_to_names_0[class_id], score)
        print(caption)
        cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), color=(0, 255, 0), thickness=2)
        cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('result.png',draw_img)
plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.show()