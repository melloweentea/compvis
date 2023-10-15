import cv2 
import sklearn 
from skimage.metrics import structural_similarity as compare_ssim
import ultralytics as YOLO
import torch

#yolo 
model = torch.hub.load("ultralytics/yolov5", "yolov5m") 
model.classes = [0]
model.conf = 0.40

#video info
# video_source = 'rtsp://admin:cctv12345@10.10.2.70:554/Streaming/Channels/1302'
video_source = 0
video = cv2.VideoCapture(video_source)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
codec = cv2.VideoWriter_fourcc(*'XVID')
fps =int(video.get(cv2.CAP_PROP_FPS))
cap_width, cap_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

name = "/Users/macbook/Desktop/code/py/ml/baksters/entech/footage/pc1.mp4"
output = cv2.VideoWriter(name, codec, fps, (cap_width, cap_height), True)
_, first_frame = video.read()

initialResults = model(first_frame)
initialObjectCount = len(initialResults.pred[0])
print("initial object count: ", initialObjectCount)

#surveillance camera
while True:
    _, frame = video.read()
    if frame is None:
        break

    #predict
    results = model(frame)
    liveObjectCount = len(results.pred[0])
    results.render()
    print("live object count: ", liveObjectCount)

    cv2.putText(frame, "no. of people: " + str(liveObjectCount), (11, 100), 0, 0.8, [0, 2550, 0], thickness=2, lineType= cv2.LINE_AA)

    if liveObjectCount > 3:
        cv2.putText(frame, "maximum exceeded", (11, 60), 0, 0.8, [0, 0, 2550], thickness=2, lineType= cv2.LINE_AA)

    cv2.imshow('person counter', frame)
    output.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output.release()
video.release()
cv2.destroyAllWindows()