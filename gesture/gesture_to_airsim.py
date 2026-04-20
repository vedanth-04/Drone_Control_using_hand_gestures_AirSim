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

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)

TARGET_Z = -3.0
client.moveToZAsync(TARGET_Z, 1).join()
client.hoverAsync().join()

print("Drone locked at altitude:", TARGET_Z)

is_landed = False
land_hold_start = None
LAND_HOLD_TIME = 2.0   # seconds

# ======================== CAMERA ========================
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

# ======================== SAFE DEMO SPEEDS ========================
MAX_SPEED = 1.8    # m/s forward/back
MAX_SIDE  = 1.2    # m/s left/right

# ======================== TILT FILTER ========================
ENTER_DEADZONE = 7
EXIT_DEADZONE  = 11
TILT_SENS = 0.045
TILT_SMOOTH = 0.15

tilt_locked = True
prev_tilt_vel = 0.0

# ======================== SPEED FILTER ========================
MIN_BEND = 90
MAX_BEND = 170
OPEN_THRESH = 0.50
SPEED_SMOOTH = 0.20
prev_speed = 0.0

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

            # -------- MODE --------
            if ext == 4 and thumb == 1:
                mode = "LAND_ARMED"
            elif ext == 4:
                mode = "FORWARD"
            elif ext == 2:
                mode = "BACKWARD"
            else:
                mode = "HOVER"

            # ================= DELAYED LANDING (2s HOLD) =================
            if mode == "LAND_ARMED" and not is_landed:
                if land_hold_start is None:
                    land_hold_start = time.time()
                elif time.time() - land_hold_start >= LAND_HOLD_TIME:
                    print("Landing after 2s open-palm hold")
                    client.landAsync().join()
                    is_landed = True
                    vx = 0
                    vy = 0
            else:
                land_hold_start = None

            # -------- TILT → SIDE --------
            wrist = lm[0]
            center = lm[9]
            raw_deg = math.degrees(math.atan2(
                center.x - wrist.x,
                -(center.y - wrist.y)
            ))

            if tilt_locked:
                if abs(raw_deg) > EXIT_DEADZONE:
                    tilt_locked = False
                    corr = raw_deg
                else:
                    corr = 0.0
            else:
                if abs(raw_deg) < ENTER_DEADZONE:
                    tilt_locked = True
                    corr = 0.0
                else:
                    corr = raw_deg

            corr = clamp(corr, -30, 30) * TILT_SENS

            if tilt_locked:
                vy = 0.0
                prev_tilt_vel = 0.0
            else:
                vy = smooth(prev_tilt_vel, corr, TILT_SMOOTH)
                prev_tilt_vel = vy

            vy = clamp(vy, -MAX_SIDE, MAX_SIDE)

            # -------- SPEED → FRONT/BACK --------
            i_ang = angle(lm[5],lm[6],lm[8])
            m_ang = angle(lm[9],lm[10],lm[12])
            avg = (i_ang + m_ang)/2

            openp = clamp((avg - MIN_BEND)/(MAX_BEND - MIN_BEND), 0, 1)

            if mode in ["FORWARD","BACKWARD"] and openp > OPEN_THRESH:
                norm = (openp - OPEN_THRESH)/(1 - OPEN_THRESH)
                target = norm
            else:
                target = 0.0

            speed = smooth(prev_speed, target, SPEED_SMOOTH)
            prev_speed = speed

            vx = speed * MAX_SPEED
            if mode == "BACKWARD":
                vx = -vx

            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

    # ================= ALTITUDE LOCKED VELOCITY =================
    if not is_landed:
        client.moveByVelocityZAsync(
            vx=vx,
            vy=vy,
            z=TARGET_Z,
            duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )

    # ================= SPEED IN KM/H =================
    speed_mps = math.sqrt(vx*vx + vy*vy)
    speed_kmph = speed_mps * 3.6

    # ================= HUD =================
    if ret:
        cv2.putText(frame,f"MODE: {mode}",(20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.putText(frame,f"Forward Vx: {vx:.2f} m/s",(20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame,f"Side Vy: {vy:.2f} m/s",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame,f"Speed: {speed_kmph:.2f} km/h",(20,130),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(frame,f"Altitude Z: {TARGET_Z}",(20,160),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(frame,f"STATUS: {'LANDED' if is_landed else 'FLYING'}",(20,190),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,"ESC = EXIT",(450,450),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.imshow("AirSim Gesture Demo (FINAL)",frame)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.04)

# ================= SAFE EXIT =================
client.hoverAsync().join()
client.enableApiControl(False)
cap.release()
cv2.destroyAllWindows()
print("System stopped safely")
    