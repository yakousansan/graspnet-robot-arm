# GRASPNT

<p align="center">
  <img src="docs/3.png" width="600" alt="GRASPNT System">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%20%2B%20Linux-blue" alt="Platform">
  <img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue" alt="C++">
  <img src="https://img.shields.io/badge/PyTorch-1.x-orange" alt="PyTorch">
</p>

<p align="center">
  <a href="README.md">简体中文</a> | <b>English</b>
</p>

---

### Overview

GRASPNT is a complete 6-DoF robotic arm grasping system that integrates GraspNet deep learning inference with real-time robot execution. The system uses an Intel RealSense D435i depth camera for RGB-D perception, GraspNet for grasp pose estimation, and a RealMan ECO65-6F robotic arm for physical execution. It adopts a Python + C++ architecture where Python handles vision, inference, and planning, while C++ handles robot control and motion execution via UDP communication.

### Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Dependencies](#dependencies)
- [Environment Setup](#environment-setup)
- [Installation & Build](#installation--build)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Demo Results](#demo-results)
- [Star History](#star-history)
- [License](#license)

---

### System Architecture

```text
D435i RGB-D
  -> Python RealSense capture
  -> Python workspace preview (human confirms target in workspace)
  -> Python requests current end-effector pose from C++ via UDP
  -> Python GraspNet inference -> grasp candidates in camera frame
  -> Python selects best candidate
  -> Python hand-eye transform + gripper TCP offset compensation
  -> Python generates base-frame pre_grasp / grasp / lift poses
  -> Python visualization + safety check
  -> Python sends grasp_execute via UDP
  -> C++ validates, prints, asks for local [y/N] confirmation
  -> C++ controls robot arm and gripper to execute grasp
```

### Features

- **6-DoF Grasp Pose Estimation** — GraspNet-based inference with collision detection, NMS, and score-based ranking
- **RealSense D435i Integration** — Aligned RGB-D capture with configurable resolution and frame rate
- **Workspace Preview** — Real-time camera view with depth overlay and workspace mask before inference
- **Hand-Eye Calibration** — Configurable camera-to-end-effector extrinsic transformation
- **Gripper TCP Compensation** — Automatic offset correction for gripper length along the grasp approach direction
- **Python + C++ Separation** — Python for vision/planning, C++ for robot control via UDP
- **Safety Checks** — Dual-layer validation: Python checks geometry/bounds, C++ checks IK reachability and robot state
- **Human-in-the-Loop** — C++ executor requires local console confirmation before executing any motion
- **Grasp Video Recording** — Automatic D435i color video recording during grasp execution
- **Rich Visualization** — 2D RGB/depth overlays, 3D point cloud with gripper wireframes, debug file export (JSON, PLY, PNG)

### Dependencies

| Component | Dependency | Purpose |
|-----------|-----------|---------|
| Python | PyTorch | GraspNet model inference |
| Python | Open3D | Point cloud processing, 3D visualization |
| Python | pyrealsense2 | Intel RealSense D435i driver |
| Python | NumPy, SciPy | Numerical computation, rotation transforms |
| Python | OpenCV | Image processing, video recording |
| Python | graspnetAPI | GraspNet data structures and evaluation |
| Python | graspnet-baseline | GraspNet model, dataset utils, collision detection |
| C++ | RealMan SDK (`api_cpp`) | RealMan ECO65-6F robot arm control |
| C++ | nlohmann/json | JSON parsing for UDP protocol |
| C++ | Winsock2 | UDP socket communication (Windows) |

### Environment Setup

> For a detailed step-by-step guide with screenshots, see the blog post: [GraspNet + D435i + RealMan 机械臂抓取环境搭建](https://blog.csdn.net/SWORDHOLDER/article/details/159793585)

**Prerequisites:**
- **Visual Studio 2019** — required for compiling `pointnet2` and `knn`. VS2022 and later versions will cause compilation errors.
- **CUDA Toolkit** — must match the PyTorch CUDA version you install.

**Python environment (Windows):**

```bash
conda create -n graspnet python=3.8
conda activate graspnet
```

**Install PyTorch** (match your CUDA version, example: CUDA 12.1):

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**GraspNet baseline:**

```bash
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
# IMPORTANT: comment out the torch line in requirements.txt first, then:
pip install -r requirements.txt
```

Compile and install the CUDA extensions:

```bash
cd pointnet2
python setup.py install
cd ../knn
python setup.py install
```

> **Note:** If you encounter `LNK2001` linker errors in `knn`, replace all `long` with `int64_t` in every file under the `knn` directory, and add `#include <cstdint>` to each file.

**graspnetAPI:**

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
# IMPORTANT: in setup.py, change dependency 'sklearn' to 'scikit-learn' before installing
pip install .
```

> **Note:** If you encounter `cannot import name 'container_abcs' from 'torch._six'`, update the imports in affected files to use `collections.abc` instead of `torch._six.container_abcs`.

**GraspNet checkpoint:**

Download the RealSense pretrained checkpoint from [Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A) and place it at `graspnet-baseline/checkpoint-rs.tar`.

Verify the installation:

```bash
cd graspnet-baseline
python demo.py --checkpoint_path checkpoint-rs.tar
```

### Installation & Build

**C++ executor (Windows, Visual Studio):**

```powershell
cd graspnt_robot_executor
mkdir build && cd build
cmake .. -DROBOTIC_ARM_DIR=path/to/Robotic_Arm
cmake --build . --config Release
```

The CMake file expects the RealMan SDK layout:

```text
graspnt_robot_executor/
  3rdparty/
    Robotic_Arm/
      include/
        rm_service.h
      lib/
        api_cpp.lib
        api_cpp.dll
```

### Usage

1. **Start the C++ executor** (connects to robot, moves to home pose, listens on UDP):

```powershell
.\graspnt_robot_executor.exe
```

2. **Start the Python grasping pipeline:**

```bash
python graspnt_rm/run_basic_grasp.py
```

3. **Workflow:**
   - Python opens a live camera preview with workspace mask
   - Press **Space** to confirm the target is in the workspace and capture the current frame
   - Python runs GraspNet inference and displays 2D/3D visualization
   - Python sends the grasp plan to C++ via UDP
   - C++ prints the plan, performs safety checks, and asks `Execute this grasp? [y/N]`
   - Type **y** to execute; the robot performs pre_grasp -> grasp -> close gripper -> lift -> home
   - Video recording is saved automatically

### Project Structure

```text
GRASPNT/
  graspnt_rm/                  # Python grasping pipeline
    run_basic_grasp.py         # Main entry point
    camera_realsense.py        # RealSense D435i capture
    graspnet_infer.py          # GraspNet inference wrapper
    transform.py               # Hand-eye transform, pose conversion
    udp_client.py              # UDP client for C++ executor
    safety.py                  # Python-side safety validation
    visualization.py           # Workspace preview + debug visualization
    video_recorder.py          # Grasp execution video recording
    config.py                  # YAML config loader
    config.yaml                # Runtime configuration

  graspnt_robot_executor/      # C++ robot execution server
    src/
      main.cpp                 # Entry point, UDP loop
      grasp_executor.cpp       # Grasp execution sequence
      protocol.cpp             # JSON protocol parsing
      robot_driver.cpp         # RealMan SDK wrapper
      safety_checker.cpp       # C++ safety validation
      udp_server.cpp           # UDP server
    include/                   # Header files

  graspnetAPI/                 # [Dependency] Official GraspNet API
  graspnet-baseline/           # [Dependency] Official GraspNet baseline model
```

### Configuration

All runtime parameters are in `graspnt_rm/config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `graspnet` | Model checkpoint, inference parameters (num_point, num_view, collision_thresh, min_score) |
| `camera` | D435i resolution and frame rate |
| `hand_eye` | Camera-to-end-effector rotation and translation from hand-eye calibration |
| `workspace` | Image region mask (center rectangle ratios) |
| `safety` | Gripper length, min grasp height, pre-grasp/lift offsets, workspace bounds |
| `execution` | UDP host/port, timeouts, retry count |
| `visualization` | Debug display mode, save options, 3D rendering parameters |
| `recording` | Video codec, frame rate, output directory |

**Important:** Before running, update the following values in `config.yaml` to match your setup:

- `graspnet.root` — path to your `graspnet-baseline` directory
- `graspnet.checkpoint` — path to your pretrained checkpoint file
- `hand_eye.rotation` / `hand_eye.translation` — your hand-eye calibration result
- `safety.gripper_length` — TCP-to-gripper-tip distance for your gripper
- `execution.udp_host` / `execution.udp_port` — must match C++ executor settings

And in `graspnt_robot_executor/src/main.cpp`:

- `robot_ip` — your RealMan controller IP address (default: `xxx.xxx.x.xx`)
- `robot_port` — your RealMan controller port (default: `8080`)
- `udp_port` — UDP listening port (default: `6556`)

### Demo Results

The following shows a real grasp execution on the ECO65-6F robotic arm with a D435i camera.

#### Grasp Execution Video

https://github.com/yakousansan/graspnet-robot-arm/raw/main/grasp_video.mp4


#### Workspace Preview

<p align="center">
  <img src="docs/1.jpeg" width="600" alt="Workspace Preview">
</p>

<p>Real-time preview before inference: RGB image with depth pseudo-color overlay, red semi-transparent mask on the region outside the workspace.</p>

#### 2D Grasp Visualization

<p align="center">
  <img src="docs/2.jpeg" width="600" alt="2D Grasp Visualization">
</p>

<p>2D RGB helper image with grasp candidate centers marked. The best candidate is a green square, other candidates are gray squares.</p>

#### 3D Point Cloud with Grasp Candidates

<p align="center">
  <img src="docs/3.png" width="600" alt="3D Grasp Visualization">
</p>

<p>3D point cloud with gripper wireframes. The best candidate is shown in green, others in gray.</p>

#### Key Execution Log

Below are the key excerpts from a real grasp run (Python side):

```text
frame: {'color_shape': (480, 640, 3), 'depth_shape': (480, 640)}
grasp: {'valid_workspace_points': 101364, 'candidate_count': 9}

score: 0.225615
width: 0.067467

pre_grasp_pose: [-0.315860, 0.099039, 0.380548, -2.894357, -0.048547, -0.063852]
grasp_pose:     [-0.309602, 0.123161, 0.283703, -2.894357, -0.048547, -0.063852]
lift_pose:      [-0.309602, 0.123161, 0.383703, -2.894357, -0.048547, -0.063852]
```

C++ executor side (UDP communication and execution):

```text
[UDP][RX] {"seq":1,"type":"pose_request","version":1}
[Executor] pose_request seq=1, reading robot state
[UDP][TX] end_pose=[-0.199184,0.183008,0.325757,3.141000,0.000000,-0.523000]
[UDP][TX] joint_deg=[149.996,0.005,90.001,0.000,-90.000,90.000]

[Executor] grasp_execute seq=2
[Command] seq=2, frame=base, unit=m_rad, score=0.225615, width=0.067467
[Command] pre_grasp_pose=[-0.315860, 0.099039, 0.380548, -2.894357, -0.048547, -0.063852]
[Command] grasp_pose=[-0.309602, 0.123161, 0.283703, -2.894357, -0.048547, -0.063852]
[Command] lift_pose=[-0.309602, 0.123161, 0.383703, -2.894357, -0.048547, -0.063852]
[UDP][TX] {"seq":2,"status":"accepted","type":"ack","version":1}
[Executor] Execute this grasp? [y/N]:
```

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yakousansan/GRASPNT&type=Date)](https://star-history.com/#yakousansan/GRASPNT&Date)

### License

This project is for research and educational purposes. Please refer to the individual dependency licenses for GraspNet, graspnetAPI, and RealMan SDK.
