import sys, numpy as np, cv2
sys.path.insert(0, 'backend')
from backend.core.tracking_autofocus_engine import (
    TrackingAutofocusEngine, _expand_bbox_to_body, _build_body_mask
)

# 1) Expansion math
bbox = (260, 80, 120, 120)
ex1, ey1, ex2, ey2 = _expand_bbox_to_body(bbox, 480, 640)
print("Tracker bbox : x=260 y=80 w=120 h=120")
print(f"Expanded body: x1={ex1} y1={ey1} x2={ex2} y2={ey2}  ({ex2-ex1}w x {ey2-ey1}h)")
assert ex2 - ex1 > 120, "Width not expanded"
assert ey2 - ey1 > 120, "Height not expanded"
assert ex1 >= 0 and ey1 >= 0 and ex2 <= 640 and ey2 <= 480, "Out of frame!"
print("Expansion : PASS")

# 2) Body mask values
mask = _build_body_mask(480, 640, ex1, ey1, ex2, ey2, feather=30)
mid_x = (ex1 + ex2) // 2
mid_y = (ey1 + ey2) // 2
print(f"Mask centre ({mid_x},{mid_y}) = {mask[mid_y, mid_x]:.2f}  (expect 1.0)")
print(f"Mask corner (0,0)            = {mask[0, 0]:.2f}  (expect 0.0)")
assert mask[mid_y, mid_x] == 1.0
assert mask[0, 0] == 0.0
print("Body mask : PASS")

# 3) End-to-end blur
e = TrackingAutofocusEngine({})
frame = np.random.randint(30, 220, (480, 640, 3), dtype=np.uint8)
cv2.rectangle(frame, (260, 60), (380, 180), (200, 60, 20), -1)

e.on_click(320, 120)
e.process_frame(frame)
out = e.process_frame(frame)
s = e.get_status()
print("State:", s["state"], " centre:", s["center"])

corner_diff = float(np.abs(frame[5, 5].astype(int) - out[5, 5].astype(int)).mean())
body_diff   = float(np.abs(frame[mid_y, mid_x].astype(int) - out[mid_y, mid_x].astype(int)).mean())
print(f"Body-centre diff (sharp side): {body_diff:.1f}")
print(f"Corner diff     (blur side)  : {corner_diff:.1f}")
print("Blur correct:", "YES" if corner_diff >= body_diff else "NO (inverted)")
pct = (np.abs(frame.astype(int) - out.astype(int)) > 8).mean() * 100
print(f"% pixels visibly changed: {pct:.1f}%  (expect >30%)")
print("\nALL TESTS PASSED" if corner_diff >= body_diff and pct > 30 else "\nSOME TESTS FAILED")
