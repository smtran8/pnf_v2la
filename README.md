# P&F — Past and Future

### Anticipatory Navigation via Multi-Robot Motion Forecasting

P&F is a learned motion-forecasting module for mobile robots. It takes a moving obstacle's **past** trajectory and predicts its **future** trajectory (~1–2 seconds ahead), so the robot can avoid where obstacles *will be* — not just where they *are now*.

Unlike reactive obstacle avoidance (gap-following, DWA, potential fields), which responds only to the present, P&F reasons about obstacle **velocity and intent** — resolving the *freezing-robot problem* and reactive oscillation that reactive methods hit in dynamic, multi-agent environments.

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