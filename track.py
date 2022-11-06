from tracker import StrongSORT
from onnx import MultiREID, Detector

import cv2
import time
import math

CLASSES = ["CAR", "MOTORCYCLE"]

def run(vid):
    
    # Initiate Detector and StongSORT
    model = Detector()
    multi_reid = MultiREID()
    strong_sort = StrongSORT(reid_model=multi_reid)
    print(f"Detector and StrongSORT has been initiated successfully")
    
    # Initiate Video object to capture
    cap = cv2.VideoCapture(vid)
    
    # Video writer to save result
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4 = cv2.VideoWriter("./output.mp4", fourcc, 30, size)
    
    prev_frame_time = 0
    new_frame_time = 0
    
    while (cap.isOpened()):
        # Get frame
        ret, ori_frame = cap.read()
        
        if ret:
            new_frame_time = time.time()
            preds = model(ori_frame)
            
            if len(preds) > 0:
                
                # If there is pred, we will update our SORT
                bboxes = preds[:, :4]
                confs = preds[:, 4]
                cls = preds[:, 5]
                time_frame = new_frame_time - prev_frame_time
                track_output = strong_sort.update(bboxes, confs, cls, ori_frame, time_frame)
                
                if len(track_output) > 0:
                    
                    for i, (out, conf) in enumerate(zip(track_output, confs)):
                        x1, y1, x2, y2 = int(out[0]), int(out[1]), int(out[2]), int(out[3])
                        id = out[4]
                        category = CLASSES[int(out[5])]
                        trajectory = out[-2]
                        
                        speed, distance = out[-1]
                        
                        time_until_hit = (distance / speed)
                        time_thresh = 5
                        
                        if category == "MOTORCYCLE":
                            theta = 30 + 5 * (2000 - distance) / 2000
                        else:
                            theta = 40 + 5 * (2000 - distance) / 2000
                        
                        
                        if (trajectory > math.cos(theta * math.pi / 180)):
                            cv2.line(ori_frame, (int(width/2), int(height)), (int(x1 + (x2 - x1) * 0.5), int(y1 + (y2 - y1) * 0.5)), (0, 0, 255), 2)
                            cv2.rectangle(ori_frame, (x1,y1), (x2, y2), (0, 0, 255), 2)
                        else:
                            cv2.line(ori_frame, (int(width/2), int(height)), (int(x1 + (x2 - x1) * 0.5), int(y1 + (y2 - y1) * 0.5)), (0, 255, 0), 2)
                            cv2.rectangle(ori_frame, (x1,y1), (x2, y2), (0, 255, 0), 2) 
                        
                        
                        cv2.putText(ori_frame, f"ID:{str(int(id))} {category}", (x1, y1 - 75), 0, 0.5, (0, 255, 0), 2)
                        cv2.putText(ori_frame, f"Trajectory:{trajectory:.3f}", (x1, y1 - 55), 0, 0.5, (0, 255, 0), 2)
                        cv2.putText(ori_frame, f"Distance:{round(distance,2)}CM", (x1, y1 - 35), 0, 0.5, (0, 255, 0), 2)
                        cv2.putText(ori_frame, f"Score:{(conf*100):.2f}%", (x1, y1 - 10), 0, 0.5, (0, 255, 0), 2)
            
            else:
                strong_sort.increment_ages()
                print("NO DETECTIONS")
                
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(ori_frame, str(int(fps)), (7, 90), 0, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
            # out_mp4.write(ori_frame)
            cv2.imshow("frame", ori_frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
            
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vid = "testvid.mp4"
    run(vid)
