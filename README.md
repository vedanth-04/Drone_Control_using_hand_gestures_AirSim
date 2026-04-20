# 🚁 Gesture-Controlled Drone Simulation using AirSim

## 📌 Overview

This project implements a **real-time hand gesture–based drone control system** using computer vision and simulation technologies. It enables intuitive, controller-free drone navigation using a single hand.

The system integrates:
- **Computer Vision (OpenCV)**
- **Hand Tracking (MediaPipe)**
- **Drone Simulation (AirSim + Unreal Engine)**

The result is a **natural Human–Machine Interface (HMI)** where gestures directly control drone motion in a realistic 3D environment.

---

## 🎯 Key Features

- ✋ Single-hand gesture control  
- 🎮 Controller-free navigation  
- 🧠 Real-time gesture recognition  
- 🛫 Stable drone simulation using AirSim  
- 📈 Smooth motion via filtering & error correction  
- 🧭 Multi-axis control (forward, lateral, yaw, altitude)  
- 🛬 Safe landing via 2-second gesture hold  
- 📊 Live telemetry (speed in km/h, altitude in meters)  

---

## ✋ Gesture Mapping

| Gesture | Action |
|--------|--------|
| 4 fingers extended | Move forward |
| 2 fingers extended | Move backward |
| Hand tilt (left/right) | Lateral movement + yaw |
| Finger bend (index + middle) | Speed control |
| Hand vertical position | Altitude control |
| Open palm (held for 2 seconds) | Land |

---

## ⚙️ How It Works

### 1. Hand Detection
MediaPipe detects and tracks **21 hand landmarks** in real time.

### 2. Gesture Recognition
Custom logic interprets:
- Finger states (open/closed)  
- Joint angles  
- Palm orientation  
- Hand position in the frame  

### 3. Motion Mapping
Gestures are converted into drone commands:
- `vx`, `vy` → horizontal movement  
- `yaw_rate` → rotation  
- `z` → altitude control  

### 4. Stabilization & Error Correction
To ensure smooth and safe control:
- Dead zones eliminate jitter  
- Smoothing filters reduce noise  
- PD-like control stabilizes altitude  
- Rate limiting prevents sudden movement 

📊 Telemetry Display

The system overlays real-time data on the camera feed:

🚀 Speed (km/h)
📏 Altitude (meters)
🎯 Current control mode
🧭 Direction feedback

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install requirements.txt

2. Setup AirSim
Install Unreal Engine
Open an AirSim environment (e.g., Blocks)
Click Play to start the simulation


3. Run the Project
python gesture_to_airsim.py
```
### 🎮 Controls Summary
- Keep hand centered → Stable hover
- Move hand up/down → Control altitude
- Tilt hand → Move left/right
- Bend fingers → Control speed
- Open palm (hold for 2 seconds) → Land safely

### ⚠️ Design Considerations
- Designed for single-hand usability
- Optimized for low-end laptops (CPU-only execution)
- Gesture mapping avoids overlap and ambiguity
- Stability is prioritized over aggressive responsiveness
- Safety limits prevent drift and sudden motion spikes

### 🔬 Future Improvements
- Dual-hand control for advanced maneuvers
- Camera orientation control
- AI-based gesture classification
- Integration with real drones (MAVLink)
- Flight data logging and replay system

### 📌 Use Cases
- Human-Drone Interaction research
- Gesture-based robotics control
- AR/VR interfaces
- Educational demonstrations
- Assistive technology applications



