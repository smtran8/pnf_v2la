# P&F — Past and Future

### Motion Forecasting for Safe Robot Navigation

P&F is a learned motion-forecasting framework for mobile robots. It takes a moving obstacle's **past** trajectory and predicts its **future** trajectory (~1–2 seconds ahead), so the robot can avoid where obstacles *will be* — not just where they *are now*.

Unlike reactive obstacle avoidance (for example gap following), which responds only to the present and stationary obstacles, P&F focuses on obstacle **velocity and intent** in environments where every agent (vehicles) move dynamically.

---

## Platform

| | |
|---|---|
| **Robot** | AgileX LIMO Pro |
| **Compute** | NVIDIA Orin Nano (onboard, real-time inference) |
| **Camera** | Orbbec Dabai depth camera (~0.3–3 m range) |
| **LiDAR** | EAI T-mini Pro (~8 m range) |
| **Middleware** | ROS2 |
| **Training** | HiPerGator |

---

## System Overview
