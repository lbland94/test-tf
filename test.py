import numpy as np
import tensorflow as tf
import cv2
# load model from path
model= tf.saved_model.load("centernet/saved_model")

img = cv2.imread("./test.jpg")
dimensions = img.shape
h = dimensions[0]
w = dimensions[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = np.expand_dims(img, 0)

# predict from model
resp = model(input_tensor)

# iterate over boxes, class_index and score list
count=0
for boxes, classes, scores in zip(resp['detection_boxes'].numpy(), resp['detection_classes'], resp['detection_scores'].numpy()):
    for box, cls, score in zip(boxes, classes, scores): # iterate over sub values in list
        if score > 0.8: # we are using only detection with confidence of over 0.8
            print(box)
            print(score)
            count += 1
            ymin = int(box[0] * h)
            xmin = int(box[1] * w)
            ymax = int(box[2] * h)
            xmax = int(box[3] * w)
            # write classname for bounding box
            # cv2.putText(img, class_names[cls], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            # draw on image
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)

# convert back to bgr and save image
cv2.imwrite("output.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(count)
