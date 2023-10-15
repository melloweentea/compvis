from ultralytics import YOLO
import cv2
import supervision as sv
from supervision.detection.utils import clip_boxes
from dataclasses import replace
import numpy as np
import math
import time

LINE_START = sv.Point(640, 0)
LINE_END = sv.Point(640,720)

def main():
    model = YOLO('yolov8m.pt')

    # src = "rtsp://admin:cctv12345@10.10.40.70:554/Streaming/Channels/0102"
    src = "/Users/macbook/Desktop/code/py/ml/baksters/entech/footage/staffhelpmove.mp4"
    video = cv2.VideoCapture(src)
    
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps =int(video.get(cv2.CAP_PROP_FPS))
    cap_width, cap_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_dir = "staffhelpmove_timed.mp4"
    output = cv2.VideoWriter(save_dir, codec, fps, (cap_width, cap_height), True)   

    #stream: used for processing long videos or live feed; uses a generator, which only
    #keeps the results of the current frame in memory, significantly reducing memory consumption.
    results = model.track(source=src, classes=0, device='cpu',
                          tracker="bytetrack.yaml", stream=True, conf=0.5)

    staff_id = None 
    people = {}
    helped = False
    start_time_recorded = False
    help_time_recorded = False

    #lower corner [left, right], upper corner [right, left]
    # polygon_coords = np.array([[640,720], [1280,720], [1280,0], [640,0]]) for webcam
    polygon_coords = np.array([[221, 110], [310, 108], [309, 3], [215, 3]]) 
    polygon_zone = sv.PolygonZone(polygon=polygon_coords, 
                                  frame_resolution_wh=(cap_width,cap_height), 
                                  triggering_position=sv.Position.CENTER)
    polygon_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.green())

    box_annotator = sv.BoxAnnotator()

    for result in results:

        frame = result.orig_img
        h, w = frame.shape[:2]
        # print(h,w)

        # frame = result.plot()

        #get the above detections automatically from sv.Detections
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            # pam's fix: detections.tracker_id = np.asarray(result.boxes.id).astype(int) 
            detections.tracker_id=result.boxes.id.cpu().numpy().astype(int)
            # cv2.putText(frame, f"ids: {detections.tracker_id}", (11, 140), 0, 0.8, [2550, 0, 0], thickness=2, lineType= cv2.LINE_AA)


        #list comprehension to return labels 
        labels = [
            f"#{track_id} {model.model.names[class_id]} {conf:0.2f}" 
            for _, _, conf, class_id, track_id
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        in_zone = polygon_zone.trigger(detections=detections) #returns if person is in the bbox
        polygon_annotator.annotate(scene=frame,label="staff zone")

        #GET CENTROID
        clipped_xyxy = clip_boxes(
            boxes_xyxy=detections.xyxy, frame_resolution_wh=(cap_width, cap_height)
        )
        clipped_detections = replace(detections, xyxy=clipped_xyxy)
        clipped_anchors = np.ceil(clipped_detections.get_anchor_coordinates(anchor=sv.Position.CENTER)).astype(int)

        # print(clipped_anchors)

        if detections.tracker_id is not None:
            if people == {}:
                for idx, id in enumerate(detections.tracker_id):
                    people[id] = [[], [], []] #position, centroid
                    people[id][0].append(in_zone[idx])
                    people[id][1] = clipped_anchors[idx]
            else: 
                for idx, id in enumerate(detections.tracker_id):
                    if id in people.keys():
                        people[id][0].append(in_zone[idx])
                        people[id][1] = clipped_anchors[idx]
                        if len(people[id][0]) == 30:
                            for i in range(10):
                                people[id][0].pop(0)
                    if id not in people.keys():
                        people[id] = [[], [], []]
                        people[id][0].append(in_zone[idx])
                        people[id][1] = clipped_anchors[idx]
                        

                for id in list(people): #cannot pop while iterating over dictionary
                    if id not in detections.tracker_id: 
                        people.pop(id)

            if staff_id is None:
                for idx, (id, info) in enumerate(people.items()):
                    if len(info[0]) > 15 and all(info[0]): 
                        staff_id = id
                        print(f"STAFF FOUND: {staff_id}")
            else:
                for idx, (id, info) in enumerate(people.items()):
                    if id == staff_id:
                        for id_, info_ in people.items():
                            if id_ != staff_id:
                                info_[2] = math.dist(info_[1], people[staff_id][1]) #calculate distance bw staff and each customer 
                                if info_[2] < 70.0:
                                    helped = True
                                print(id_, info_[2])
                            else:
                                info_[2] = None
                        if not all(info[0]):
                            cv2.putText(frame, "staff left", (11, 100), 0, 0.8, [0, 50, 2500], thickness=2, lineType= cv2.LINE_AA)
                            # staff_id = None
                    else:
                        if not helped:
                            cv2.putText(frame, "customer needs help", (11, 140), 0, 0.8, [0, 2550, 0], thickness=2, lineType= cv2.LINE_AA)
                            if not start_time_recorded:
                                start_time = time.time()
                                start_time_recorded = True 
                        else: 
                            if not help_time_recorded:
                                help_time = math.ceil(time.time() - start_time)
                                help_time_recorded = True 
                            cv2.putText(frame, f"customer helped in {help_time}s", (11, 140), 0, 0.8, [0, 2550, 0], thickness=2, lineType= cv2.LINE_AA) 

    
        # print(people)

        cv2.putText(frame, f"staff id: {staff_id}", (11, 60), 0, 0.8, [0, 2550, 0], thickness=2, lineType= cv2.LINE_AA)

        cv2.imshow('track', frame)
        output.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    output.release()
    video.release()
    cv2.destroyAllWindows

if __name__ == '__main__':
    main()
