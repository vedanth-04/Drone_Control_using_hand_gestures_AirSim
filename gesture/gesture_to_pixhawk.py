import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import time
import math
import airsim

# ======================== AIRSIM CONNECTION ========================
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim")

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)

TARGET_Z = -3.0
client.moveByVelocityZAsync(0, 0, TARGET_Z, 2).join()
client.hoverAsync().join()
print("Drone hovering at fixed start altitude")

# ======================== CAMERA SETUP ========================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(5, 25)

# ======================== MEDIAPIPE ========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

# ======================== SAFE MOTION LIMITS ========================
MAX_SPEED = 1.6         # m/s  (~5.7 km/h)  noticeable but safe
MAX_SIDE  = 1.4         # m/s
MAX_YAW   = 18          # deg/sec (slow & stable)

# ======================== TILT FILTER ========================
ENTER_DEADZONE = 7
EXIT_DEADZONE  = 11
TILT_SENS = 0.35
TILT_SMOOTH = 0.14

tilt_locked = True
prev_tilt = 0.0

# ======================== SPEED FILTER ========================
MIN_BEND = 90
MAX_BEND = 170
OPEN_THRESH = 0.45
SPEED_SMOOTH = 0.25
prev_speed = 0.0

# ======================== ALTITUDE CONTROL (HAND Y) ========================
ALT_CENTER_Y = 0.5
ALT_DEADZONE  = 0.08
ALT_KP = 0.9
ALT_KD = 0.25
ALT_RATE_LIMIT = 0.15

MIN_Z = -1.5
MAX_Z = -6.0

current_z = TARGET_Z
prev_alt_error = 0.0

# ======================== LANDING CONTROL ========================
LAND_HOLD_TIME = 2.0
land_start_time = None

# ======================== UTILS ========================
def clamp(x, a, b):
    return max(a, min(b, x))

def smooth(prev, new, alpha):
    return (1 - alpha) * prev + alpha * new

def finger_states(lm):
    tips = [8,12,16,20]
    fingers = [1 if lm[t].y < lm[t-2].y else 0 for t in tips]
    thumb = 1 if lm[4].x < lm[3].x else 0
    return fingers, thumb

def angle(a,b,c):
    ba = [a.x-b.x, a.y-b.y]
    bc = [c.x-b.x, c.y-b.y]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(clamp(dot/mag, -1, 1)))

# ======================== MAIN LOOP ========================
while True:
    ret, frame = cap.read()

    vx = 0.0
    vy = 0.0
    yaw_rate = 0.0
    mode = "HOVER"

    if ret:
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark
            fingers, thumb = finger_states(lm)
            ext = sum(fingers)

            # ---------------- MODE ----------------
            if ext == 4 and thumb == 0:
                mode = "FORWARD"
            elif ext == 2:
                mode = "BACKWARD"
            elif ext == 5:
                mode = "LAND"
            else:
                mode = "HOVER"

            # ---------------- LAND HOLD ----------------
            if mode == "LAND":
                if land_start_time is None:
                    land_start_time = time.time()
                elif time.time() - land_start_time > LAND_HOLD_TIME:
                    print("Landing initiated by gesture")
                    client.landAsync().join()
                    break
            else:
                land_start_time = None

            # ---------------- TILT → SIDE + YAW ----------------
            wrist = lm[0]
            center = lm[9]
            dx = center.x - wrist.x
            dy = center.y - wrist.y
            raw = math.degrees(math.atan2(dx,-dy))

            if tilt_locked:
                if abs(raw) > EXIT_DEADZONE:
                    tilt_locked = False
                    corr = raw
                else:
                    corr = 0.0
            else:
                if abs(raw) < ENTER_DEADZONE:
                    tilt_locked = True
                    corr = 0.0
                else:
                    corr = raw

            corr = clamp(corr,-30,30) * TILT_SENS

            if tilt_locked:
                roll = 0.0
                prev_tilt = 0.0
            else:
                roll = smooth(prev_tilt,corr,TILT_SMOOTH)
                prev_tilt = roll

            vy = clamp(roll, -1, 1) * MAX_SIDE
            yaw_rate = clamp(roll, -1, 1) * MAX_YAW

            # ---------------- SPEED FROM FINGER BEND ----------------
            i_ang = angle(lm[5],lm[6],lm[8])
            m_ang = angle(lm[9],lm[10],lm[12])
            avg = (i_ang+m_ang)/2

            openp = clamp((avg-MIN_BEND)/(MAX_BEND-MIN_BEND),0,1)

            if mode in ["FORWARD","BACKWARD"] and openp > OPEN_THRESH:
                norm = (openp-OPEN_THRESH)/(1-OPEN_THRESH)
                target = norm
            else:
                target = 0.0

            speed = smooth(prev_speed,target,SPEED_SMOOTH)
            prev_speed = speed

            vx = speed * MAX_SPEED
            if mode == "BACKWARD":
                vx = -vx

            # ---------------- ALTITUDE (HAND UP/DOWN) ----------------
            wrist_y = lm[0].y
            alt_error = ALT_CENTER_Y - wrist_y

            if abs(alt_error) < ALT_DEADZONE:
                alt_error = 0.0

            alt_velocity = (ALT_KP * alt_error) + (ALT_KD * (alt_error - prev_alt_error))
            alt_velocity = clamp(alt_velocity, -ALT_RATE_LIMIT, ALT_RATE_LIMIT)

            current_z += alt_velocity * 0.1
            current_z = clamp(current_z, MAX_Z, MIN_Z)
            prev_alt_error = alt_error

            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

    # ---------------- SEND FLIGHT COMMAND ----------------
    client.moveByVelocityZAsync(
        vx=vx,
        vy=vy,
        z=current_z,
        duration=0.1,
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
    )

    speed_kmph = abs(vx) * 3.6

    # ---------------- HUD ----------------
    if ret:
        cv2.putText(frame,f"MODE: {mode}",(20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.putText(frame,f"Speed: {speed_kmph:.1f} km/h",(20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(frame,f"Yaw Rate: {yaw_rate:.1f}",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame,f"Altitude: {-current_z:.2f} m",(20,130),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.putText(frame,"Move hand UP/DOWN for height",(20,165),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),2)
        cv2.putText(frame,"Open palm 2s = LAND",(20,195),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),2)
        cv2.putText(frame,"ESC = EXIT",(470,460),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.imshow("FINAL AIRSIM GESTURE DEMO",frame)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.04)

# ======================== SAFE EXIT ========================
client.hoverAsync().join()
client.enableApiControl(False)
cap.release()
cv2.destroyAllWindows()
print("System stopped safely")
