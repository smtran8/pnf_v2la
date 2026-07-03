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

---
## Step 2 — Metric (Real World) Projection

The second step is to plot the real x,y world coordinates from any moving LIMO. 
General Idea:
- In each LIMO's car, the camera will return some information, including:
  + A Depth Distance: This is the distance along the camera's forward axis - which is the Z coordinate. This distance reveals how far the object is, but in the camera direction. A visual example is when an object is located to the very left side of the camera.
  + Intrinsic Paramters: For example fx and fy (focal length). These are verical and horizontal to convert pixels to pure meters. For example, if a LIMO shifts 1 meter distance, what will be the change in pixel and vice versa.
  + Principal Point: cx and cy. Optical center of the image - used to calculate an image offset from the center.
  
For a camera:
- Z = points forward, out the lens, the direction the camera looks. This is depth.
- X = points sideways (left/right across the image).
- Y = points up/down (vertical).
  
**Formula:**

X = (u - cx) * Z/fx

In which:
- X is the LIMO's side way distance
- cx is the principal point of the image, fx is the focal length defined
- Z is the Camera Depth Distance
- u is the moving agent (another LIMO)'s pixel location, which is probably the bottom-center of the bounding box.

The output will be a tuple of (Z, X) which shows what is the distance for Z and how much side way for X
**Why not use LiDAR?**
The formula above aims to solve a question - how far away is a LIMO in a real world. But a LiDAR can already do that, and can do much better, so why am I choosing the solution above? The answer is that a LiDAR can identify the distance better, but the detector and tracker cannot run on it. Note that detector and tracker only runs on image/video, so the camera with constant images will be a good suit for detector and tracker. Note: There is still a way to use LiDAR, especially for longer distance (~8 meters) rather than the camera depth with only 3 meters - but this upgrade will be implemented later.

**June 19th)** Metric Projection works, however at this step I have not implemented the detector/tracker yet. Two observations that could be fixed by using detector and tracker:
- The camera is default to have a very straightforward ray, and it sits on the head of the LIMO. Therefore, it will not see the other LIMO but rather just the wall. Lift the LIMO up will allow the distance to be correct.
- The side information will need the data of the bounding box from the detector.

**Update**:
- Because the bounding box can contain Non-Limo objects, most of the times will be the wall behind, which could inflate the distance => Apply a minimum window options. For every pixel, take all values in the surrounding pixels (decided by a window size) and choose the minimum value => To differentiate between Limo and a wall.
- Also to get the most accurate distance data, for any time a LIMO camera depth glitch (between None and the real distance value), we will revert back to the previous distance. Because LIMO moves slowly, the distance difference is not significant, allowing enough time for the real distance to show up.

## Step 3 - Gather data and build Forecasting Model
**June 23th**:
-This is a Time-Series forecasting - we define input/output as windows:
- Given each time step containing Time Stamp with Forward Distance and Side Distance (calculated above), a window is a collection of time steps, with a defined size. Assume we have 25Hz frequency, a window with size = 25 will cover 1 second of movement.
- Our Prediction will be calculating 1 second of output, with 1 second of given input. For example, assume we have a window of size 50. Assume both our input and output window has size 25. In the first index slicing, intput window will take data from timestep 0 to 24, and the output window will take data from timestep 25 to 49. At 25Hz, input/output window both convey 1 second, so we have a pipeline of 1 second of input/1 second of output. 
**Baseline Model**:
- For the baseline comparison, we pick a constant velocity formula.
+ Velocity = (Position[t] - Position [t-1]) / dt with t = a specific time}
+ Predicted Position: Position[t + k] = Position [t] + Velocity * (k + 1) * dt
- This constant velocity is expected to have high errors, as a LIMO robot will drive non-linearly, with curving and short stops

## Step 4 - Build LSTM and Transformer Model
- Completed building LSTM (2 Encoder and 2 Decoder Layer, Dropout 0.1, Adam Optimizer, Learning Rate 0.001, MSE Loss Function, Teacher Forcing 0.5) - 0.17m Average Displacement Error
- Completed building Transformer (2 Encoder and 2 Decoder Layer, 4 attention heads, 64 Dimensions for model, Feed-Forward Network is 256 dimensions, Dropout 0.1, Adam Optimizer, Learning Rate 0.001, MSE Loss Function) - 0.19m Average Displacement Error

## Step 5 - Showcase
- Deploy the LSTM model on NVIDIA Jetson Orin Nano. Implement Projection on 2D axis to better see predicted trajectory. Video: [Link](https://drive.google.com/file/d/1l3VpN5fp3n_ECLokBzg0-T79yaTTh0Ih/view?usp=sharing)
- Finished project.
