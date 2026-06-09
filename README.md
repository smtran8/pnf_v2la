# P&F — Past and Future

### Motion Forecasting for Safe Robot Navigation

P&F is a learned motion-forecasting framework for mobile robots. It takes a moving obstacle's **past** trajectory and predicts its **future** trajectory (~1–2 seconds ahead), so the robot can avoid where obstacles *will be* — not just where they *are now*.

Unlike reactive obstacle avoidance (for example gap following), which responds only to the present and stationary obstacles, P&F focuses on obstacle **velocity and intent** in environments where every agent (vehicle) moves dynamically.

---

## Platform

| | |
|---|---|
| **Robot** | AgileX LIMO Pro | Supported by V2LA lab @UF
| **Compute** | NVIDIA Jetson Orin Nano |
| **Camera** | Orbbec Dabai depth camera (~0.3–3 m range) |
| **LiDAR** | EAI T-mini Pro (~8 m range) |
| **Middleware** | ROS2 |
| **Training** | HiPerGator |

---

## System Overview
The main model (P&F model) will sometimes be referred to as forecaster. The forecaster will consume outputs from the detector and tracker, which will be introduced below. Obstacles in this testing environment are **other LIMO rovers** in the lab. Note that obstacles will also be referred to as vehicles or agents.

---
## Step 1 — Detector & Tracker

The first component is the perception front-end: it turns raw camera frames into **persistent, identified tracks** that P&F model will later consume. 

### Detector
- Run an object detector (YOLO-class) on the LIMO camera stream.
- **Will possibly fine-tune to recognize a LIMO** 
- **Goal:** reliably detect (through putting a box) a LIMO across varied distances, angles, and lighting.
- **June 5th** - Tried YOLO26 by Ultralytics: [https://docs.ultralytics.com/tasks/detect#export](Reference). The model cannot detect a LIMO => Start gathering data and will fine-tune model.
- **June 6th** - Tried YOLO26 again with light fine-tuning (100 epochs, batch = 8, patience = 20) with 200+ labeled image. Achieve great result but suspect potential data leakage (duplicate images) => Try again when gets to lab. Unseen image could reveal if the model actually overfits.
- **June 9th** - Done Detector. Achived a model with great confidence: 90% with static image and 70% with live video.
### Tracker
- Add a multi-object tracker (ByteTrack) on top of the detector.
- The tracker assigns a **persistent track ID** to each individual agent, maintained across frames. Reference:(https://blog.roboflow.com/what-is-bytetrack-computer-vision/)
- **Class ID - limo vs. Track ID - an Int (1,2,3):** class ID is set to limo; the *track ID* is the unique identity of one agent over time (limo 1, limo 2). Forecasting model needs the track ID to assemble one trajectory per agent.
- **Possible Challenge:** *ID switches* — identities being swapped after two agents cross or briefly occlude.
- **June 9th** Done Tracker. The ID switches limit happens when a LIMO goes behind another LIMO, disappearing in the frames and gets a new ID when appears again. This challenge should not affect the final forecasting model; if a LIMO disappears in the frame, there is no way to predict its trajectory.


