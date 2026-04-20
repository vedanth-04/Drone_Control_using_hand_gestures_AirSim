import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*"
)

import cv2
import mediapipe as mp
import time
import math
import airsim   # <-- ADDED

# ===================== AIRSIM SETUP =====================
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")
print(f"Client Ver:{client.getClientVersion()}")

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)

# read current altitude as our target (z is negative upwards)
state = client.getMultirotorState()
TARGET_Z = state.kinematics_estimated.position.z_val
print(f"Hover altitude (Z): {TARGET_Z:.2f}")

# very conservative base throttle
BASE_THROTTLE = 0.60
MIN_THR = 0.52
MAX_THR = 0.68
ALT_KP  = 0.06   # small proportional gain to keep near TARGET_Z

# maximum angles in radians (VERY SMALL for slow demo)
MAX_PITCH_RAD = 0.05   # ~2.8 deg
MAX_ROLL_RAD  = 0.04   # ~2.3 deg

# ===================== CAMERA SETUP =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FPS, 25)

if not cap.isOpened():
    raise Exception("Camera failed to open.")
print("Camera opened successfully")

# ===================== MEDIAPIPE SETUP =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

# ===================== TILT (ROLL) PARAMETERS =====================
ENTER_DEADZONE_DEG = 7.0       # snap-to-zero threshold
EXIT_DEADZONE_DEG  = 11.0      # must exceed this to move again
TILT_MAX_DEG = 30.0
TILT_SENSITIVITY = 0.5
TILT_SMOOTH_ALPHA = 0.12

prev_tilt_out = 0.0
tilt_locked_zero = True

# ===================== SPEED (FINGER BEND) PARAMETERS =====================
MIN_BEND_DEG = 90.0
MAX_BEND_DEG = 170.0
OPENNESS_ACTIVATE = 0.40      # must open fingers at least this much
MIN_SPEED_SCALE = 0.0
MAX_SPEED_SCALE = 1.0
BEND_SMOOTH_ALPHA = 0.20

prev_speed_scale = 0.0

# ===================== UTILITY FUNCTIONS =====================
def clamp(x, a, b):
    return max(a, min(b, x))

def smooth(prev, new, alpha):
    return (1 - alpha) * prev + alpha * new

def get_finger_states(lm):
    tips = [8, 12, 16, 20]
    fingers = []
    for tip in tips:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    thumb = 1 if lm[4].x < lm[3].x else 0
    return fingers, thumb

def finger_bend_angle(a, b, c):
    ba = [a.x - b.x, a.y - b.y]
    bc = [c.x - b.x, c.y - b.y]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba * mag_bc == 0:
        return 0.0
    cos_ang = clamp(dot / (mag_ba * mag_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_ang))

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mode = "NO HAND"
    raw_tilt = 0.0
    tilt_out = 0.0
    tilt_label = "CENTER"

    idx_angle = 0.0
    mid_angle = 0.0
    avg_bend = 0.0
    openness = 0.0
    speed_scale = 0.0
    speed_label = "OFF"
    speed_percent = 0

    # default commands (hover)
    pitch_cmd = 0.0
    roll_cmd  = 0.0

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        fingers, thumb = get_finger_states(lm)
        ext = sum(fingers)

        # ---------- MODE DETECTION ----------
        if ext == 4 and thumb == 1:
            mode = "EMERGENCY (OPEN PALM)"
            # we will just hover for demo, no disarm
        elif ext == 4:
            mode = "FORWARD"
        elif ext == 2:
            mode = "BACKWARD"
        elif ext == 0 and thumb == 0:
            mode = "ROTATE"
        else:
            mode = "HOVER / IDLE"

        # ========== HARD SNAP TILT WITH HYSTERESIS ==========
        wrist = lm[0]
        mid_base = lm[9]

        dx = mid_base.x - wrist.x
        dy = mid_base.y - wrist.y
        raw_tilt = math.degrees(math.atan2(dx, -dy))

        if tilt_locked_zero:
            if abs(raw_tilt) > EXIT_DEADZONE_DEG:
                tilt_locked_zero = False
                corrected_tilt = raw_tilt
            else:
                corrected_tilt = 0.0
        else:
            if abs(raw_tilt) < ENTER_DEADZONE_DEG:
                tilt_locked_zero = True
                corrected_tilt = 0.0
            else:
                corrected_tilt = raw_tilt

        corrected_tilt = clamp(corrected_tilt, -TILT_MAX_DEG, TILT_MAX_DEG)
        corrected_tilt *= TILT_SENSITIVITY

        if tilt_locked_zero:
            tilt_out = 0.0
            prev_tilt_out = 0.0
        else:
            tilt_out = smooth(prev_tilt_out, corrected_tilt, TILT_SMOOTH_ALPHA)
            prev_tilt_out = tilt_out

        if abs(tilt_out) < 3:
            tilt_label = "CENTER"
        else:
            side = "LEFT" if tilt_out < 0 else "RIGHT"
            mag = abs(tilt_out)
            if mag < 10:
                lvl = "SLIGHT"
            elif mag < 20:
                lvl = "MEDIUM"
            else:
                lvl = "STRONG"
            tilt_label = f"{side} ({lvl})"

        # ========== SPEED FROM FINGER BEND ==========
        idx_angle = finger_bend_angle(lm[5], lm[6], lm[8])
        mid_angle = finger_bend_angle(lm[9], lm[10], lm[12])
        avg_bend = (idx_angle + mid_angle) / 2.0

        openness = (avg_bend - MIN_BEND_DEG) / (MAX_BEND_DEG - MIN_BEND_DEG)
        openness = clamp(openness, 0.0, 1.0)

        if mode in ["FORWARD", "BACKWARD"]:
            if openness < OPENNESS_ACTIVATE:
                target_speed = 0.0
            else:
                active_norm = (openness - OPENNESS_ACTIVATE) / (1.0 - OPENNESS_ACTIVATE)
                active_norm = clamp(active_norm, 0.0, 1.0)
                target_speed = MIN_SPEED_SCALE + active_norm * (MAX_SPEED_SCALE - MIN_SPEED_SCALE)
        else:
            target_speed = 0.0

        speed_scale = smooth(prev_speed_scale, target_speed, BEND_SMOOTH_ALPHA)
        prev_speed_scale = speed_scale

        speed_percent = int(speed_scale * 100 + 0.5)

        if speed_scale < 0.05:
            speed_label = "OFF"
        elif speed_scale < 0.35:
            speed_label = "SLOW"
        elif speed_scale < 0.7:
            speed_label = "CRUISE"
        else:
            speed_label = "FAST"

        # -------- MAP CAMERA LOGIC TO DRONE ANGLES --------
        # speed_scale -> pitch (forward/back)
        if mode == "FORWARD":
            pitch_cmd = +speed_scale * MAX_PITCH_RAD
        elif mode == "BACKWARD":
            pitch_cmd = -speed_scale * MAX_PITCH_RAD
        else:
            pitch_cmd = 0.0

        # tilt_out -> roll (LEFT/RIGHT), independent of speed magnitude
        max_tilt_scaled = TILT_MAX_DEG * TILT_SENSITIVITY
        if max_tilt_scaled > 0:
            tilt_norm = clamp(tilt_out / max_tilt_scaled, -1.0, 1.0)
        else:
            tilt_norm = 0.0
        roll_cmd = tilt_norm * MAX_ROLL_RAD

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        prev_tilt_out = smooth(prev_tilt_out, 0.0, TILT_SMOOTH_ALPHA)
        prev_speed_scale = smooth(prev_speed_scale, 0.0, BEND_SMOOTH_ALPHA)
        tilt_out = prev_tilt_out
        speed_scale = prev_speed_scale
        mode = "NO HAND"
        tilt_label = "CENTER"
        speed_label = "OFF"
        speed_percent = int(speed_scale * 100 + 0.5)
        pitch_cmd = 0.0
        roll_cmd  = 0.0

    # ===================== HUD (EVALUATOR FRIENDLY) =====================
    cv2.putText(frame, f"MODE: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"TILT: {tilt_label}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Tilt Value: {tilt_out:6.2f} deg", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    cv2.putText(frame, f"Finger Openness: {int(openness*100)}%", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    cv2.putText(frame, f"Speed: {speed_percent:3d}% [{speed_label}]", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ----- SPEED BAR -----
    bar_x, bar_y, bar_w, bar_h = 10, 190, 300, 22
    filled_w = int(bar_w * clamp(speed_scale, 0.0, 1.0))

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

    cv2.putText(frame, "0%", (bar_x, bar_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "100%", (bar_x + bar_w - 50, bar_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "ESC = Exit", (500, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("FULL GESTURE TEST : Tilt + Speed + Mode + AirSim", frame)

    # -------- SIMPLE ALTITUDE HOLD AROUND TARGET_Z --------
    state = client.getMultirotorState()
    z_now = state.kinematics_estimated.position.z_val
    alt_error = TARGET_Z - z_now
    throttle_corr = ALT_KP * alt_error
    throttle = clamp(BASE_THROTTLE + throttle_corr, MIN_THR, MAX_THR)

    # -------- SEND COMMAND TO AIRSIM (angle + throttle) --------
    try:
        client.moveByAngleThrottleAsync(
            pitch_cmd,
            roll_cmd,
            throttle,
            0.0,        # yaw_rate
            duration=0.08
        )
    except Exception as e:
        print("AirSim command error:", e)
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
client.hoverAsync().join()
client.enableApiControl(False)
print("Full gesture + AirSim control stopped safely.")
