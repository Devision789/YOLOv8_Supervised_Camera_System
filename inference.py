import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import cv2
model = YOLO("yolov8n.pt")
image_count = 0
'''polygons = np.array([
    [150, 400], [500, 400], [500, 80], [350, 80]
]) 1.mp4 '''
'''polygons = np.array([
    [100, 230], [200, 600], [580, 250], [400, 150]
]) 3.mp4'''

polygons = np.array([
    [10, 300], [500, 300], [400, 100], [200, 100]
])
zones = sv.PolygonZone(polygon=polygons)   

box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator()
zone_annotator = sv.PolygonZoneAnnotator(zone=zones, color=sv.Color.BLUE, thickness=6)
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global image_count
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    #detections = detections[detections.class_id == 0]
    mask = zones.trigger(detections=detections)
    detections = detections[(detections.class_id == 0) & mask]
    
    
    
    labels = [
    model.model.names[class_id]
    for class_id in detections.class_id
]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)
    
        
    for xyxy in detections.xyxy:
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
            #sink.save_image(image=cropped_image)
        image_name = f"ketrom2_{image_count}.png"
        image_path = os.path.join('outputs', image_name)
        cv2.imwrite(image_path, cropped_image)
        image_count += 1
    return frame

sv.process_video(
    source_path="data/2.mp4",
    target_path="outputs/trom2.mp4",
    callback=callback
)