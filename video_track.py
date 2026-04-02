import cv2
from ultralytics import YOLO
import math
import csv
import matplotlib.pyplot as plt



video_path = r"C:\Users\aldawlia\Desktop\New folder\nnm.mp4"
model = YOLO("yolov8n.pt")

distance_threshold = 50          # pixels (for ID matching)
exit_time_threshold = 2.0        # seconds
period_duration = 300            # 5 minutes

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

people = {}        #(cx, cy)
people_time = {}   # id -> time data
next_id = 0

frame_id = 0
crowded_time = []
period_stats = {}
max_people_seen = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 360))

    frame_id += 1
    current_time = frame_id / fps

    results = model(frame, conf=0.6, classes=[0])
    current_people_count = sum(len(r.boxes) for r in results)

    detected_ids = set()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2

            matched_id = None
            min_dist = distance_threshold

            for pid, point in people.items():
                dist = distance((cx, cy), point)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = pid

            if matched_id is not None:
                people[matched_id] = (cx, cy)
                person_id = matched_id
            else:
                people[next_id] = (cx, cy)
                person_id = next_id
                next_id += 1

            detected_ids.add(person_id)

            if person_id not in people_time:
                people_time[person_id] = {
                    "start_time": current_time,
                    "last_time": current_time,
                    "exited": False,
                    "exit_time": None
                }
            else:
                people_time[person_id]["last_time"] = current_time

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f'ID {person_id}', (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

   
    for pid, t in people_time.items():
        if not t["exited"]:
            if current_time - t["last_time"] > exit_time_threshold:
                t["exited"] = True
                t["exit_time"] = t["last_time"]

    #Frame analytics
    active_people = sum(
        1 for t in people_time.values() if not t["exited"]
    )

    max_people_seen = max(max_people_seen, active_people)
    crowded_time.append((current_time, active_people))

    current_period = int(current_time // period_duration)
    if current_period not in period_stats:
        period_stats[current_period] = {"total_people": 0, "frames": 0}

    period_stats[current_period]["total_people"] += active_people
    period_stats[current_period]["frames"] += 1

    cv2.putText(frame, f'People: {current_people_count}', (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.putText(frame, f'Max Seen: {max_people_seen}',
                 (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.putText(frame, f'Time: {current_time:.2f}s', (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow("People Analytics", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# RESULTS 
print("--------------------------------------------")
for pid, t in people_time.items():
    total = t["last_time"] - t["start_time"]
    print(f"Person {pid}: {total:.2f} sec")

empty_periods = []
start_empty = None

for time, count in crowded_time:
    if count == 0:
        if start_empty is None:
            start_empty = time
    else:
        if start_empty is not None:
            empty_periods.append((start_empty, time))
            start_empty = None

for s, e in empty_periods:
    print(f"From {s:.2f}s to {e:.2f}s")

peak_period = None
peak_avg = 0

for period, stats in period_stats.items():
    avg = stats["total_people"] / stats["frames"]
    if avg > peak_avg:
        peak_avg = avg
        peak_period = period

if peak_period is not None:
    start = peak_period * period_duration
    end = start + period_duration
    print(f"Peak: {start:.1f}s → {end:.1f}s | Avg: {peak_avg:.2f}")



with open("summary.csv",'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Person ID", "start_time","end_time","duration_seconds"])

    for pid, tiems in people_time.items():
        end_time = tiems['exit_time'] if tiems['exit_time'] is not None else tiems['last_time']
        duration = end_time - tiems['start_time']

        writer.writerow([
            pid,
            round(tiems['start_time'],2),
            round(end_time,2),
            round(duration,2)
        ])

with open("crowd_time.csv",'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time_sec", "active_people"])

    for time, count in crowded_time:
        writer.writerow([time, count])

with open("peak_period.txt",'w') as f:
    f.write(f"Peak Period: {peak_period if peak_period is not None else 'None'}\n")
    f.write(f"Peak Average: {peak_avg if peak_avg > 0 else 0}\n")


tiems = [t for t,_ in crowded_time]
counts = [c for _,c in crowded_time]

plt.figure(figsize=(10,5))
plt.plot(tiems, counts, label="Active People", color='blue')
plt.xlabel("Time (seconds)")
plt.ylabel("Number of People")
plt.title("Crowd Density Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()




cap.release()
cv2.destroyAllWindows()
