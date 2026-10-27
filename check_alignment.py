import sys, cv2, pandas as pd

folder, set_no, video = sys.argv[1], sys.argv[2], sys.argv[3]
offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0

shots = pd.read_csv(f"data/CoachAI-Projects/ShuttleSet/set/{folder}/set{set_no}.csv")
shots = shots.dropna(subset=["hit_x", "hit_y"])          # serves have no hit point
picks = shots.iloc[[0, len(shots)//3, 2*len(shots)//3, -1]]  # 4 shots spread out

cap = cv2.VideoCapture(video)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
scale = w / 1280                                          # labels are in 1280x720
print(f"video fps={cap.get(cv2.CAP_PROP_FPS):.2f}, width={w:.0f}, scale={scale}")

for i, s in picks.iterrows():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s.frame_num) + offset)
    ok, frame = cap.read()
    if not ok:
        print("could not read frame", s.frame_num); continue
    x, y = int(s.hit_x * scale), int(s.hit_y * scale)
    cv2.circle(frame, (x, y), 12, (0, 0, 255), 3)
    cv2.putText(frame, f"rally {s.rally} shot {int(s.ball_round)} {s.type}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(f"check_{i}.jpg", frame)
    print("saved", f"check_{i}.jpg", "frame", int(s.frame_num))
