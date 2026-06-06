# GRASPNT Python + C++ 抓取框架说明

本文档说明当前已经跑通的 GRASPNT + D435i + ECO65-6F 抓取系统。它的目标不是只告诉你怎么运行，而是解释清楚每个模块为什么存在、输入输出是什么、数据怎样从相机坐标变成机械臂动作，以及出现问题时应该从哪里排查。

## 1. 系统目标

系统采用 Python 和 C++ 分工：

- Python 负责 D435i 图像采集、GRASPNT/GraspNet 推理、抓取候选可视化、抓取计划生成、基础数据检查、UDP 发送抓取计划。
- C++ 负责连接 RealMan ECO65-6F 机械臂、读取机械臂当前位姿、Modbus 夹爪控制、最终安全检查、本地人工确认和真实运动执行。

这样划分的原因是：

- GRASPNT 推理依赖 PyTorch、Open3D、GraspNet API、点云处理，Python 环境更适合。
- 机械臂 SDK、Modbus、Windows 上的稳定运动控制，在 C++ 中更可靠。
- Python 只输出“抓取建议”，C++ 决定“是否安全执行”和“是否真的运动”。

整体数据流：

```text
D435i RGB-D
  -> Python RealSense 采集
  -> Python workspace 预览，人工确认目标在工作区内
  -> Python 向 C++ 请求当前机械臂末端位姿
  -> Python GRASPNT 推理，得到 camera 坐标系下的抓取候选
  -> Python 选择最优候选
  -> Python 做手眼变换和夹爪 TCP 偏移补偿
  -> Python 生成 base 坐标系下的 pre_grasp / grasp / lift
  -> Python 可视化和基础安全检查
  -> Python 通过 UDP 发送 grasp_execute
  -> C++ 解析、打印、检查、等待本地 y 确认
  -> C++ 控制机械臂和夹爪执行抓取
```

## 2. 当前完整运行流程

真实运行时的顺序如下：

```text
1. 启动 C++ graspnt_robot_executor
2. C++ 连接 192.168.1.20:8080 的机械臂控制器
3. C++ MoveJ 到初始关节姿态 [150, 0, 90, 0, -90, 90] deg
4. C++ 监听 UDP 6556
5. 启动 Python run_basic_grasp.py
6. Python 启动 D435i
7. Python 打开带 workspace 遮罩的实时预览
8. 用户确认目标在工作区内，按 Space 固定当前 RGB-D 帧
9. Python 发送 pose_request
10. C++ 读取机械臂当前状态，回复 pose_response
11. Python 用固定 RGB-D 帧执行 GRASPNT 推理
12. Python 得到抓取候选，过滤、排序，选择第 1 个作为最优抓取
13. Python 把最优抓取从 camera 坐标系转换到 base 坐标系
14. Python 生成 pre_grasp_pose、grasp_pose、lift_pose
15. Python 显示 2D/3D 可视化结果，打印抓取计划
16. Python 做基础安全检查
17. Python 开始保存 D435i 彩色视频
18. Python 发送 grasp_execute
19. C++ 打印解析后的抓取计划
20. C++ 做协议、安全、可达性检查
21. C++ 回复 ack accepted
22. C++ 控制台询问 Execute this grasp? [y/N]
23. 输入 y 时 C++ 执行动作；其他输入返回 cancelled，不运动
24. C++ 执行完成后返回 result
25. Python 停止录像并保存视频
26. Python 打印 execution_result
```

## 3. Python 端代码结构

Python 包目录：

```text
graspnt_rm/
  camera_realsense.py
  config.py
  config.yaml
  graspnet_infer.py
  run_basic_grasp.py
  safety.py
  transform.py
  udp_client.py
  visualization.py
```

### 3.1 `run_basic_grasp.py`

这是 Python 主入口，负责串联完整流程。它不直接控制机械臂，也不直接控制夹爪。

核心步骤：

```python
config = load_config(config_path)
validate_runtime_config(config)
camera.start()
frame = preview_workspace(...)
robot_state = udp_client.request_pose()
candidates, grasp_report = runner.infer(...)
plan = build_plan(...)
visualize_debug(...)
validate_motion_plan(...)
execution_result = udp_client.execute_grasp(plan)
```

重要设计点：

- preview 阶段只看相机画面，不推理，不请求机械臂运动。
- 只有按 Space 确认当前画面后，Python 才固定这一帧 RGB-D。
- `pose_request` 放在 preview 之后，是因为 D435i 装在末端，如果预览期间移动机械臂，之前的末端位姿会失效。
- Python 生成合法计划后总是发送 `grasp_execute`，是否真实运动由 C++ 控制台确认。
- 如果没有候选、位姿不合法、低于安全高度、超出 workspace，Python 会拦截，不发送给 C++。

### 3.2 `camera_realsense.py`

负责 D435i 采集。

输入：

```yaml
camera:
  width: 640
  height: 480
  fps: 30
```

输出 `RGBDFrame`：

```python
color       # BGR 彩色图，shape = (H, W, 3)，uint8
depth       # 深度图，shape = (H, W)，uint16
intrinsics  # width/height/fx/fy/cx/cy/scale
depth_scale # RealSense 深度比例
```

内部做了这些事：

- 开启 color stream，格式为 `bgr8`
- 开启 depth stream，格式为 `z16`
- 用 `rs.align(rs.stream.color)` 把 depth 对齐到 color
- 读取 RealSense 深度比例 `depth_scale`
- 返回对齐后的 RGB-D 和相机内参

### 3.3 `visualization.py`

负责预览、调试图和 3D 抓取可视化。

当前有两个阶段的可视化：

1. 抓取前 workspace preview
2. 推理后的候选抓取 debug visualization

workspace preview：

- 显示实时 RGB 图
- 显示深度伪彩色图
- 对 workspace 外区域叠加红色半透明遮罩
- 按 Space 继续
- 按 q 或 Esc 取消

推理后可视化：

- 2D：RGB 图上标记抓取中心，最优候选为绿色，其他候选为灰色
- Depth：显示伪彩色深度图
- 3D：Open3D 显示点云和候选夹爪线框
- Debug 文件：保存 RGB/Depth 图、3D 抓取候选截图、候选 JSON、点云 PLY

`debug_outputs` 中常见文件含义：

- `*_rgb_grasp.png`：2D RGB 辅助图，只在图像上标记候选抓取中心；最优候选是绿色方块，其他候选是灰色方块。
- `*_depth.png`：深度伪彩色图，用来判断目标深度是否缺失、是否有异常跳变。
- `3d_grasps.png` / `3d_grasps_001.png`：3D 点云 + 多个候选夹爪线框截图；最优候选为绿色夹爪，其他候选为灰色夹爪。这是最接近 Open3D 可视化窗口的离线图片。命名不带时间戳，已有文件不会被覆盖，后续结果会自动递增编号。
- `*_grasps.json`：候选抓取的平移、旋转矩阵、分数、开口宽度，以及最终发送给 C++ 的 `pre_grasp_pose/grasp_pose/lift_pose`。
- `*_scene.ply`：当前场景点云，可用 Open3D、CloudCompare 等工具离线打开。
- `videos/grasp_video.avi` / `videos/grasp_video_001.avi`：抓取执行阶段的 D435i 彩色视频。命名不带时间戳，已有文件不会被覆盖，后续结果会自动递增编号。

可视化非常重要，因为它能帮助判断：

- 目标是否在 workspace 内
- 深度图是否有效
- GraspNet 候选是否在正确物体上
- 最优候选姿态是否合理
- 坐标转换后的抓取计划是否明显异常

### 3.4 `video_recorder.py`

负责抓取过程录像。

当前版本为了简化协议，不等待 C++ 输入 `y` 后再开始录，而是在 Python 侧确认抓取计划安全后、发送 `grasp_execute` 前开始录制。这样不需要新增 UDP 事件，也不需要修改 C++。

录像时间范围：

```text
开始：validate_motion_plan() 通过之后，udp_client.execute_grasp(plan) 之前
结束：收到 C++ result 之后；如果 UDP 执行异常，也会在 finally 中停止录像
```

优点：

- 实现简单，不需要 C++ 新增 `motion_started` 协议。
- 视频包含机械臂运动前的初始画面，便于复盘。
- 即使 C++ 端取消或执行失败，视频也会保存下来。

代价：

- 如果 C++ 等待你输入 `y` 的时间较长，视频开头会多录一段等待画面。

## 4. GRASPNT / GraspNet 推理部分

### 4.1 推理输入

`GraspNetRunner.infer()` 的输入是：

```python
color             # D435i 彩色图，BGR，uint8
depth             # D435i 深度图，uint16
intrinsics        # 相机内参
workspace_config  # 图像工作区配置
```

配置示例：

```yaml
graspnet:
  root: "D:/ArmProject/GraspNet/graspnet-baseline"
  checkpoint: "D:/ArmProject/GraspNet/graspnet-baseline/checkpoint-rs.tar"
  num_point: 30000
  num_view: 300
  collision_thresh: 0.01
  voxel_size: 0.01
  min_score: 0.05
  top_down_angle_deg: 45
  approach_axis: 0
```

其中：

- `root`：graspnet-baseline 根目录，用于加载 `models`、`dataset`、`utils`
- `checkpoint`：RealSense 训练权重
- `num_point`：输入网络的点数量
- `num_view`：网络视角采样数，需要和 checkpoint 兼容
- `collision_thresh`：模型无关碰撞检测阈值
- `voxel_size`：碰撞检测体素大小
- `min_score`：最低抓取置信度
- `top_down_angle_deg`：优先保留接近自上而下的候选
- `approach_axis`：抓取姿态旋转矩阵中哪一列作为接近方向

### 4.2 RGB-D 到点云

代码使用 GraspNet baseline 的：

```python
create_point_cloud_from_depth_image(depth, camera, organized=True)
```

相机内参转换为：

```python
CameraInfo(
    width,
    height,
    fx,
    fy,
    cx,
    cy,
    scale,
)
```

深度点转三维点的基本原理是：

```text
z = depth / scale
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

这里的三维点在 D435i camera 坐标系下。

### 4.3 workspace mask

workspace 用来限制网络只看目标区域，减少桌面、背景、机械臂结构的干扰。

当前支持：

```yaml
workspace:
  mode: "center"
  x_min_ratio: 0.2
  x_max_ratio: 0.8
  y_min_ratio: 0.2
  y_max_ratio: 0.8
```

`build_workspace_mask()` 的逻辑：

- 先保留 `depth > 0` 的像素
- 如果 `mode=center`，再保留图像中心矩形区域
- mask 为 `True` 的点进入推理

如果 workspace 内有效点数为 0，Python 直接报错，不进入网络推理。

### 4.4 点云采样

网络输入点数固定为 `num_point`。

如果 workspace 点数大于 `num_point`：

```text
随机无放回采样 num_point 个点
```

如果 workspace 点数小于 `num_point`：

```text
保留全部点，再随机有放回补足 num_point
```

这样网络始终拿到固定大小的点云张量：

```python
point_clouds: shape = (1, num_point, 3)
```

### 4.5 网络推理

推理过程：

```python
with torch.no_grad():
    end_points = self.net(end_points)
    grasp_preds = self.pred_decode(end_points)
```

网络输出通过 `pred_decode()` 解码成一组 6D 抓取候选。每个候选包含：

```python
translation      # 抓取中心，camera 坐标系，单位 m
rotation_matrix  # 抓取姿态，camera 坐标系下的 3x3 矩阵
score            # 抓取质量分数
width            # 夹爪张开宽度，单位 m
```

这些候选还不是机械臂能直接执行的位姿，因为它们在 camera 坐标系下。

### 4.6 碰撞检测、NMS、排序

推理后执行：

```python
gg = gg[~collision_mask]
gg = gg.nms()
gg = gg.sort_by_score()
```

含义：

- 碰撞检测：过滤掉夹爪模型和点云明显碰撞的候选
- NMS：去掉空间上高度重叠、姿态相近的重复候选
- sort_by_score：按抓取质量分数排序

然后转换成项目内部的 `GraspCandidate`：

```python
GraspCandidate(
    translation=np.asarray(grasp.translation),
    rotation_matrix=np.asarray(grasp.rotation_matrix),
    score=float(grasp.score),
    width=float(grasp.width),
)
```

### 4.7 候选过滤和最优选择

`filter_grasp_candidates()` 做两类过滤：

1. 分数过滤：

```python
candidate.score >= min_score
```

2. 自上而下倾向过滤：

```python
approach = candidate.rotation_matrix[:, approach_axis]
angle = arccos(dot(approach, [0, 0, 1]))
angle <= top_down_angle_deg
```

如果存在满足 top-down 条件的候选，就只保留这批候选；如果没有，则回退使用分数过滤后的候选。

最终 `run_basic_grasp.py` 使用：

```python
candidates[0]
```

作为最优候选。因为前面已经按 score 排序，所以第一个候选是当前策略下的最高分候选。

### 4.8 推理输出报告

推理返回：

```python
candidates
grasp_report
```

`grasp_report` 例子：

```python
{
  "valid_workspace_points": 100972,
  "candidate_count": 10
}
```

这些数字用于判断：

- workspace 内点数是否足够
- 候选数量是否正常
- 如果候选太少，可能是 workspace 错、深度缺失、物体不明显、阈值过高或碰撞检测过严

## 5. 坐标系和位姿转换

### 5.1 涉及的坐标系

当前系统涉及：

- camera：D435i 相机坐标系
- end：机械臂末端坐标系
- base：机械臂基座坐标系
- grasp：GraspNet 输出的抓取坐标系
- gripper：本项目定义的夹爪/TCP 执行坐标系

最终发送给 C++ 的位姿必须是：

```text
base 坐标系，单位 m_rad，格式 [x, y, z, rx, ry, rz]
```

### 5.2 手眼外参

配置：

```yaml
hand_eye:
  direction: "camera_to_end"
  rotation: [...]
  translation: [...]
```

含义是：

```text
end_from_camera = H_cam2end
```

也就是点从 camera 坐标系变换到 end 坐标系：

```text
P_end = H_cam2end * P_camera
```

### 5.3 当前变换链

完整链路：

```text
base_from_gripper =
    base_from_end
  * end_from_camera
  * camera_from_grasp
  * grasp_from_gripper
```

其中：

- `base_from_end`：由 C++ 返回的当前末端位姿构造，目前 Python 使用 `end_pose`
- `end_from_camera`：手眼标定外参
- `camera_from_grasp`：GRASPNT 候选在相机坐标系下的位置和姿态
- `grasp_from_gripper`：GraspNet 抓取姿态到实际夹爪 TCP 姿态的对齐和偏移

### 5.4 `end_pose` 和 `H_end2base`

C++ 返回两种末端位姿表达：

```json
"end_pose": [x, y, z, rx, ry, rz]
"H_end2base": [[...], [...], [...], [...]]
```

当前 Python 使用 `end_pose`：

```python
base_from_end = pose_to_transform(current_end_pose)
```

你的测试中 `end_pose` 重建出来的矩阵和 `H_end2base` 很接近，所以不是前面 x 大偏差的主因。后续为了更稳，可以让 Python 优先使用 `H_end2base`，减少欧拉角约定带来的潜在风险。

### 5.5 GraspNet 姿态和夹爪姿态对齐

GraspNet 输出的 rotation matrix 不是机械臂可直接执行的夹爪姿态。项目使用：

```python
DEFAULT_GRASPNET_TO_GRIPPER = [
    [ 0, 0, 1],
    [ 0, 1, 0],
    [-1, 0, 0],
]
```

它表示：

```text
grasp_from_gripper.R = DEFAULT_GRASPNET_TO_GRIPPER
```

也就是把本项目的夹爪执行坐标系对齐到 GraspNet 抓取坐标系。

### 5.6 夹爪长度 / TCP 偏移补偿

配置：

```yaml
safety:
  gripper_length: 0.18
```

这个值表示：

```text
机械臂末端 TCP 到实际抓取接触点的距离，单位 m
```

它解决的问题是：GraspNet 给的是物体上的抓取点，但机械臂运动控制的是 TCP。如果不补偿，机械臂会把 TCP 移到物体抓取点，夹爪实体会撞过目标。

正确补偿逻辑在 `transform.py`：

```python
gripper_offset_in_grasp =
    align_rotation @ [0, 0, -gripper_length]

grasp_from_gripper =
    Transform(
        R = align_rotation,
        t = gripper_offset_in_grasp
    )
```

为什么不能直接写：

```python
[0, 0, -gripper_length]
```

因为 `[0, 0, -gripper_length]` 必须先定义在夹爪自身坐标系，再转换到 GraspNet 抓取坐标系。如果直接写在 GraspNet 坐标系下，就会沿错轴补偿。你之前测试到 x 偏差约 17 cm，就是这个问题导致的。

验证现象：

```text
gripper_length = 0.18 时，旧逻辑 x 约 -0.47
gripper_length = 0.00 时，x 回到约 -0.30
修正补偿方向后，gripper_length 可恢复真实值
```

### 5.7 输出姿态格式

最终姿态：

```python
[x, y, z, rx, ry, rz]
```

其中：

- `x/y/z`：base 坐标系下的位置，单位 m
- `rx/ry/rz`：欧拉角，单位 rad
- C++ 按 `rm_pose_t.position` 和 `rm_pose_t.euler` 发送给机械臂 SDK

## 6. 抓取计划生成

### 6.1 `grasp_pose`

`grasp_pose` 是真正闭合夹爪的位姿：

```text
grasp_pose = camera_grasp_to_base_pose(best_candidate)
```

它已经包含：

- GraspNet candidate translation
- GraspNet candidate rotation
- GraspNet 到夹爪姿态对齐
- gripper_length / TCP 偏移补偿
- 手眼外参
- 当前机械臂末端到基座变换

### 6.2 `pre_grasp_pose`

`pre_grasp_pose` 是抓取前的接近位姿：

```python
pre_grasp_pose = offset_pose_along_local_z(
    grasp_pose,
    -pre_grasp_offset
)
```

含义：

```text
沿抓取姿态自己的局部 Z 方向后退
```

这样夹爪从物体外侧沿抓取方向接近目标，而不是从任意方向撞过去。

### 6.3 `lift_pose`

`lift_pose` 是抓取后抬升位姿：

```python
lift_pose = offset_pose_along_base_z(
    grasp_pose,
    lift_offset
)
```

含义：

```text
保持 x/y/rx/ry/rz 不变，只把 base 坐标系 z 增加 lift_offset
```

这样能避免沿抓取局部方向“抬升”时反而向下压。这个问题之前已经出现过：局部 +Z 朝下时，`lift_pose.z` 会比 `grasp_pose.z` 更低。

### 6.4 三个位姿的运动意义

```text
pre_grasp_pose:
  夹爪打开，从安全距离接近目标

grasp_pose:
  真正抓取点，夹爪在这里闭合

lift_pose:
  抓取后沿 base +Z 抬起，离开桌面和周围障碍
```

## 7. Python 安全检查

Python 侧只做基础数据有效性和几何边界检查。

当前检查：

- pose 必须是 6 维
- 每个值必须是有限数，不能是 NaN 或 inf
- 每个 pose 的 z 必须高于 `min_grasp_z`
- 每个 pose 必须在 `workspace_bounds` 内
- 没有候选时直接报错
- workspace 内没有有效深度点时直接报错

Python 不做：

- 机械臂逆解是否可达的最终判断
- 机械臂是否报警
- 夹爪是否正常
- 是否允许真实运动

这些由 C++ 负责。

## 8. UDP 协议

### 8.1 `pose_request`

Python 发送：

```json
{
  "version": 1,
  "type": "pose_request",
  "seq": 1
}
```

C++ 返回：

```json
{
  "version": 1,
  "type": "pose_response",
  "seq": 1,
  "status": "ok",
  "frame": "base",
  "unit": "m_rad",
  "end_pose": [-0.199, 0.183, 0.326, 3.141, 0.0, -0.523],
  "joint_deg": [150, 0, 90, 0, -90, 90],
  "H_end2base": [[...], [...], [...], [...]]
}
```

### 8.2 `grasp_execute`

Python 发送：

```json
{
  "version": 1,
  "type": "grasp_execute",
  "seq": 2,
  "command_id": "20260604_104142_0002",
  "frame": "base",
  "unit": "m_rad",
  "pre_grasp_pose": [-0.32, 0.19, 0.20, -3.04, 0.23, 2.77],
  "grasp_pose": [-0.30, 0.18, 0.11, -3.04, 0.23, 2.77],
  "lift_pose": [-0.30, 0.18, 0.21, -3.04, 0.23, 2.77],
  "score": 0.20,
  "width": 0.056
}
```

C++ 先返回：

```json
{
  "version": 1,
  "type": "ack",
  "seq": 2,
  "command_id": "20260604_104142_0002",
  "status": "accepted"
}
```

执行后返回：

```json
{
  "version": 1,
  "type": "result",
  "seq": 2,
  "command_id": "20260604_104142_0002",
  "status": "success"
}
```

如果 C++ 本地未输入 y：

```json
{
  "version": 1,
  "type": "result",
  "seq": 2,
  "command_id": "...",
  "status": "cancelled",
  "reason": "operator declined"
}
```

### 8.3 `seq` 和 `command_id`

- `seq` 用于区分请求和响应，防止收到旧包。
- `command_id` 用于区分一次抓取命令，防止重复执行。
- C++ 日志中隐藏 `command_id`，但协议中仍保留它。

## 9. C++ 端代码结构

C++ 项目目录：

```text
graspnt_robot_executor/
  CMakeLists.txt
  README.md
  include/
    grasp_executor.h
    protocol.h
    robot_driver.h
    safety_checker.h
    udp_server.h
  src/
    grasp_executor.cpp
    main.cpp
    protocol.cpp
    robot_driver.cpp
    safety_checker.cpp
    udp_server.cpp
```

### 9.1 `main.cpp`

负责：

- 创建 `RobotDriver`
- 连接机械臂
- 启动时 MoveJ 到 home：

```text
[150, 0, 90, 0, -90, 90] deg
```

- 启动 UDP server
- 接收 JSON
- 打印分行日志
- 调用 protocol 解析命令
- 调用 `GraspExecutor` 检查和执行
- 在执行前询问：

```text
Execute this grasp? [y/N]
```

只有输入 `y` 或 `Y` 才运动。

### 9.2 `protocol.cpp`

负责协议解析和响应构造。

解析 `grasp_execute` 时检查：

- JSON 必须是 object
- `type` 必须是 `grasp_execute`
- `command_id` 必须存在
- `pre_grasp_pose` 必须是 6 个数字
- `grasp_pose` 必须是 6 个数字
- `lift_pose` 必须是 6 个数字
- 如果 Python 发送了 `home_joint_deg`，也必须是 6 个数字

### 9.3 `robot_driver.cpp`

封装 RealMan SDK。

主要接口：

```cpp
Connect(ip, port)
Disconnect()
CurrentState(state)
MoveJ(joints_deg, speed)
MoveL(pose, speed)
TargetReachable(pose)
ConfigureModbus()
OpenGripper()
CloseGripper()
```

实现细节：

- `Connect()` 调用 `rm_init(RM_TRIPLE_MODE_E)` 和 `rm_create_robot_arm`
- `CurrentState()` 调用 `rm_get_current_arm_state`
- `ReadJointDegree()` 读取当前关节角
- `FillCurrentEnd2Base()` 使用正解得到当前 `H_end2base`
- `MoveJ()` 调用 `rm_movej`
- `MoveL()` 调用 `rm_movel`
- `TargetReachable()` 调用 `rm_algo_inverse_kinematics`
- 夹爪通过 Modbus 寄存器写目标位置

### 9.4 `grasp_executor.cpp`

负责真正执行一次抓取。

执行顺序：

```text
Validate(command)
MoveJ(home_joint)
ConfigureModbus
OpenGripper
MoveL(pre_grasp_pose)
MoveL(grasp_pose)
CloseGripper
MoveL(lift_pose)
MoveJ(home_joint)
```

这样保证：

- 启动时回 home
- 每次抓取开始前回 home
- 抓取完成后回 home

### 9.5 `safety_checker.cpp`

C++ 侧最终安全检查。

当前检查：

- `version == 1`
- `frame == "base"`
- `unit == "m_rad"`
- 速度在范围内
- home joint 有限
- pre/grasp/lift 三个位姿有限
- z 高于 C++ 最小高度
- x/y/z 在 C++ workspace 范围内

`GraspExecutor::Validate()` 还会调用：

```cpp
robot_.TargetReachable(...)
```

对三个位姿做逆解可达性检查。

## 10. 配置说明

### 10.1 `graspnet`

```yaml
graspnet:
  root: "D:/ArmProject/GraspNet/graspnet-baseline"
  checkpoint: "D:/ArmProject/GraspNet/graspnet-baseline/checkpoint-rs.tar"
  num_point: 30000
  num_view: 300
  collision_thresh: 0.01
  voxel_size: 0.01
  min_score: 0.05
  top_down_angle_deg: 45
  approach_axis: 0
```

调参建议：

- 候选太少：降低 `min_score`，临时关闭或降低 `collision_thresh`
- 推理慢：降低 `num_point`
- 抓取姿态过斜：减小 `top_down_angle_deg`
- 目标被背景干扰：缩小 workspace

### 10.2 `camera`

```yaml
camera:
  width: 640
  height: 480
  fps: 30
```

D435i 必须保证 depth 和 color 都正常，深度缺失会直接影响点云和抓取点位置。

### 10.3 `hand_eye`

```yaml
hand_eye:
  direction: "camera_to_end"
  rotation: [...]
  translation: [...]
```

这部分来自手眼标定。任何旋转方向、平移单位、矩阵是否取逆的问题，都会直接影响 base 下的抓取位置。

### 10.4 `workspace`

```yaml
workspace:
  mode: "center"
  x_min_ratio: 0.2
  x_max_ratio: 0.8
  y_min_ratio: 0.2
  y_max_ratio: 0.8
```

workspace 只影响图像区域和点云输入，不等同于机械臂 base 坐标系的 workspace bounds。

### 10.5 `safety`

```yaml
safety:
  gripper_length: 0.18
  min_grasp_z: 0.02
  pre_grasp_offset: 0.10
  lift_offset: 0.10
  workspace_bounds:
    x: [-0.8, 0.8]
    y: [-0.8, 0.8]
    z: [0.01, 0.9]
```

含义：

- `gripper_length`：TCP 到抓取接触点的距离，单位 m
- `min_grasp_z`：三个位姿最低 z 限制
- `pre_grasp_offset`：抓取前沿局部方向后退距离
- `lift_offset`：抓取后沿 base +Z 抬升距离
- `workspace_bounds`：base 坐标系下允许发送给 C++ 的范围

### 10.6 `visualization`

```yaml
visualization:
  enabled: true
  mode: "both"
  top_n: 20
  save_debug: true
  debug_dir: "debug_outputs"
  save_3d_grasp_image: true
  save_3d_visible: false
  save_point_cloud: true
```

含义：

- `enabled`：是否启用预览、显示和保存。调试抓取时建议保持 `true`。
- `mode`：`both` 同时显示 2D 和 3D；`save_only` 只保存文件不弹出调试显示。
- `top_n`：最多显示/保存前 N 个候选。第 1 个候选就是当前代码选择的最优抓取。
- `save_debug`：是否保存调试文件。
- `debug_dir`：调试文件输出目录。
- `save_3d_grasp_image`：是否保存 `3d_grasps.png` / `3d_grasps_001.png` 这类图片。这张图包含点云和候选夹爪线框，最优抓取为绿色；命名不带时间戳，已有文件不会被覆盖，后续结果会自动递增编号。
- `save_3d_visible`：截图时 Open3D 临时窗口是否可见。默认 `false`；如果 Windows 上保存出的 3D 图片为空白，可以改成 `true`。
- `save_point_cloud`：是否额外保存 `*_scene.ply` 点云。

### 10.7 `recording`

```yaml
recording:
  enabled: true
  output_dir: "debug_outputs/videos"
  filename_stem: "grasp_video"
  extension: ".avi"
  codec: "MJPG"
  fps: 30
  stop_timeout_sec: 2.0
```

含义：

- `enabled`：是否保存抓取过程视频。
- `output_dir`：视频保存目录。
- `filename_stem`：视频基础文件名。
- `extension`：视频扩展名。第一版推荐 `.avi`。
- `codec`：OpenCV fourcc 编码。Windows 上推荐先用 `MJPG`，通常比 `mp4v` 更稳定。
- `fps`：写入视频的目标帧率，建议和 D435i color fps 保持一致。
- `stop_timeout_sec`：停止录像时等待后台采集线程退出的最长时间。

保存规则：

```text
debug_outputs/videos/grasp_video.avi
debug_outputs/videos/grasp_video_001.avi
debug_outputs/videos/grasp_video_002.avi
```

已有文件不会被覆盖。

## 11. 常见问题和排查方法

### 11.1 可视化看不到目标

优先检查：

- 物体是否在 workspace 遮罩内
- 深度图是否有有效深度
- D435i 是否被遮挡或反光影响
- `workspace` 比例是否太小

### 11.2 GraspNet 返回候选很少

可能原因：

- workspace 内有效点太少
- `min_score` 太高
- `collision_thresh` 太严格
- 物体太平、透明、反光或深度缺失
- 预训练模型对当前物体泛化不好

### 11.3 姿态看起来对，但位置偏

按顺序排查：

1. `gripper_length` 是否正确
2. 夹爪长度补偿方向是否正确
3. 手眼外参是否是 `camera_to_end`
4. 外参单位是否是 m
5. `end_pose` 和 `H_end2base` 是否一致
6. D435i 深度是否准确
7. GraspNet 抓取点是否在物体中心还是物体边缘

你已经验证过的关键问题：

```text
gripper_length = 0.18 且补偿方向错误时，x 偏到约 -0.47
gripper_length = 0 时，x 回到约 -0.30
修正补偿方向后，夹爪长度可以恢复真实值
```

### 11.4 lift_pose 反而向下

原因：

```text
沿抓取局部 +Z 抬升时，局部 +Z 可能朝 base -Z
```

当前已经修正：

```text
lift_pose 沿 base +Z 抬升
```

### 11.5 Python 没有发送 grasp_execute

可能原因：

- 没有候选
- `validate_motion_plan()` 不通过
- 位姿低于 `min_grasp_z`
- 超出 `workspace_bounds`
- UDP `pose_response` 无效
- preview 被取消

### 11.6 C++ 收到后不运动

可能原因：

- C++ 控制台没有输入 `y`
- C++ safety checker 拒绝
- IK 不可达
- 机械臂 SDK 返回错误
- Modbus 配置失败
- 夹爪开合失败

## 12. 当前已跑通状态

目前系统已经完成：

- C++ 能连接 ECO65-6F
- C++ 启动后能回 home
- Python 能采集 D435i RGB-D
- Python 能预览 workspace
- Python 能完成 GRASPNT 推理和可视化
- Python 能生成 base 坐标系三段抓取位姿
- Python 能通过 UDP 发送给 C++
- C++ 能解析、打印、ack
- C++ 本地确认后能执行抓取
- 抓取完成后能回 home
- 夹爪长度补偿方向问题已定位并修复
- lift_pose 抬升方向已改为 base +Z

## 13. 后续可优化点

1. Python 优先使用 `H_end2base`，减少欧拉角重建误差。
2. 将 C++ 的 home_joint、speed、workspace bounds 放入 C++ 配置文件。
3. 增加 C++ 机械臂状态检查，如错误码、急停、当前模式。
4. 增加夹爪实际状态反馈，而不只写目标寄存器。
5. 增加多候选策略：如果第 1 个候选不可达，自动尝试第 2、第 3 个候选。
6. 增加固定测试物体和标定板验证流程，用于量化 camera->base 位置误差。
7. 增加日志文件保存，方便复盘每次抓取的 RGB-D、候选、位姿和执行结果。
