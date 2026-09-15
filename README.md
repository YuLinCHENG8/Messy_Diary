# 三连杆三维工作空间示意

这个示例用 Python、NumPy 和 Matplotlib 描绘一个三连杆串联机构的三维工作空间，并显示可交互姿态。左侧点云是关节角限制内、末端执行器能到达的位置集合。

## 模型约定

默认连杆长度为：

```text
L1 = 100
L2 = 20
L3 = 30
```

三个连杆在各自局部坐标系的正 z 轴方向延伸，采用右手坐标系和主动旋转。三个关节角及限制分别是：

```text
q1: 绕 base frame 的 z 轴，范围 [-180°, 180°]
q2: 绕第一级旋转后的局部 y 轴，范围 [-40°, 40°]
q3: 绕前两级旋转后的局部 x 轴，范围 [-110°, 110°]
```

因此旋转顺序是：

```text
R = Rz(q1) @ Ry(q2) @ Rx(q3)
```

各段位置为：

```text
p1 = Rz(q1) @ [0, 0, L1]
p2 = p1 + Rz(q1) @ Ry(q2) @ [0, 0, L2]
p3 = p2 + Rz(q1) @ Ry(q2) @ Rx(q3) @ [0, 0, L3]
```

这里 `p3` 是末端位置，`R` 是末端 frame 相对于 base frame 的姿态。

如果你的第二、第三个旋转轴始终固定在 base frame，而不是随前级 frame 一起运动，公式会不同；当前代码采用的是串联机构中更常见的“前级局部轴”解释。

## 运行

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python three_link_workspace.py
```

默认按关节分别密采样末端可达点：`q1` 步长由 `--step` 控制（默认 10°），`q2`/`q3` 自动更密，从而得到连续的可达点云而不是稀疏壳层。可以调整基础采样间隔：

```bash
python three_link_workspace.py --step 5
```

只计算并打印示例位姿、不打开绘图窗口：

```bash
python three_link_workspace.py --no-plot
```

打开绘图窗口后，底部有 `q1`、`q2`、`q3` 三个滑块和对应的数值输入框。拖动滑块或输入角度并按回车，右侧机构姿态会更新；输入超出限制的数值会自动限制到允许范围内。
