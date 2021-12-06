import cv2
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import tensorflow as tf


MODEL = "yolo\yolov3-face.cfg"
WEIGHT = "yolo\yolov3-wider_16000.weights"

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

data_folder = 'portrait_group3'
Model_Path = 'my_face_model'
class_names = ["Duc_Anh","Hang",'Hoi','Ngoc','Yen']

model = tf.keras.models.load_model(Model_Path)

IMG_WIDTH, IMG_HEIGHT = 416, 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


#import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # frame is now the image capture by the webcam (one frame of the video)
    #cv2.imshow('Input', frame)
#--------------------------------------------------------------------

    IMG_WIDTH, IMG_HEIGHT = 416, 416

# Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 
							 1/255, (IMG_WIDTH, IMG_HEIGHT),
                             [0, 0, 0], 1, crop=False)

# Set model input
    net.setInput(blob)

# Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

# Run 'prediction'
    outs = net.forward(output_layers)
    
#--------------------------------------------------------------------
#The blob result will have the shape of (1, 3, IMG_WIDTH, IMG_HEIGHT). 
#In order to view it, use the below code snippet

    blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3]) 
    print(blobb.shape)

    #plt.figure(figsize=(15, 15))
    #plt.imshow(blobb, cmap='gray')
    #plt.axis('off')
    #plt.show()

#--------------------------------------------------------------------

    frame_height = frame.shape[0]

    frame_width = frame.shape[1]

# Scan through all the bounding boxes output from the network and keep only
# the ones with high confidence scores. Assign the box's class label as the
# class with the highest score.

    confidences = []
    boxes = []

    
# Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
		# One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
				# Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
            
						# Find the top left point of the bounding box 
                topleft_x = int(center_x - width/2) 
                topleft_y = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

# Perform non-maximum suppression to eliminate 
# redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#--------------------------------------------------------------------

    final_boxes = []
    result = frame.copy()

    for i in indices:
        box = boxes[i]
        final_boxes.append(box)

    # Extract position data
        left = box[0]    #x
        top = box[1]     #y
        width = box[2]   #w
        height = box[3]  #h

    # Draw bouding box with the above measurements
    ### YOUR CODE HERE
        cv2.rectangle(result, (left, top), (left + width, top + height), (0, 255, 0), 2)
		
		# Display text about confidence rate above each box
        text = f'{confidences[i]:.2f}'
        ### YOUR CODE HERE
        cv2.putText(result,text,(left,top),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)


    
    result2 = frame.copy()

    resize_new = cv2.resize(result2,(299,299))
    expand = np.expand_dims(resize_new,axis = 0)
    prediction = model.predict(expand)
    pred_in = np.argmax(prediction,axis=-1)
    label = class_names[pred_in[0]]


  

# Display text about number of detected faces on topleft corner
# YOUR CODE HERE
    cv2.putText(result, f'Number of faces detected {len(final_boxes)}',(20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(result, label + " ", (left, top -35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)    
# frame is now the image capture by the webcam (one frame of the video)

    cv2.imshow('face detection', result)

    c = cv2.waitKey(1)
		# Break when pressing ESC
    if c == 27:
        break
    #elif c==ord("s"):
        #crop = frame[top:(top+height), left:(left+width)]
        #cv2.imwrite(f'auto_cam/{i:03d}.jpg',crop)
        #i+=1

cap.release()
cv2.destroyAllWindows()



