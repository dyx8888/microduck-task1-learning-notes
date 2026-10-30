# 【Datawhale MicroDuck 组队学习】Task7：RDK X5 与多策略部署 · BPU 验收学习笔记

## 我这次在学什么

前六期解决「怎么训出一个会动的策略」，这一期解决「训好的策略怎么真的在板子上跑起来」。

不过读完 `07_RDK网页与多策略_BPU验收.ipynb` 和板端调度脚本之后，我发现这一期真正的难点不在转换命令，而在两件事：**怎么证明这段视频是本次板端算出来的**，以及**「支持多策略切换」这句话到底切的是什么**。

先说清楚本次的取证口径：我手上没有 RDK X5，也没有在 Ubuntu 上重跑过。下面所有数字都来自仓库里**已保存的单元输出**（GPU 报告 `run_id=92aa7fcd…`、`generated_at_utc=2026-10-11T08:57:59Z`），以及我从 `rdk_bpu_policy_supervisor.py` 读到的实际代码。凡是推断的地方我都会标明。

---

## 一、ONNX、HBM、BPU 各自服务于谁

| 角色 | 服务于谁 | 本质 | 在仓库里的位置 |
| --- | --- | --- | --- |
| ONNX | 训练侧 → 部署工具链 | 与硬件无关的计算图 + 权重 | `microduck_bpu_models/converted/*.opset11.onnx` |
| HBM | 板端 BPU 运行时 | 编译后的指令流 + 量化权重 | 工作站 `compiled/.../model_output/*.bin` → 板端 `/home/sunrise/microduck_policy/hbm/*.bin` |
| BPU | 机器人本体 | 执行指令流，输出 14 维动作 | RDK X5 上由 `rdk_bpu_policy_server.py` 驱动 |

有两个细节值得记下来，它们和直觉不一样：

1. **编译产物的后缀是 `.bin`，不是 `.hbm`。** `BPU_HBM_DEFAULTS` 里每一条都指向 `model_output/xxx.bin`（例如 `walking/model_output/walking.opset11.bin`）。大家口头说「HBM」，指的是这类编译产物的格式，不靠文件后缀区分。
2. **Notebook 里的 `--hbm` 参数是工作站本地路径，不是板端路径。** 它只用来读文件算 SHA256、再切片上传；真正被加载的是板端的 `/home/sunrise/microduck_policy/hbm/walking.bin`。看到命令行里有 `--hbm` 就以为「HBM 是在工作站上跑的」，是很容易踩的误读。

### 为什么要经过 ONNX → HBM 这一跳

这一步不是改后缀，而是至少四件性质不同的事：

- **算子落地**：BPU 只支持 ONNX 算子集的一部分，不支持的要么被替换，要么回落到 CPU。后者会直接破坏推理时延的确定性——对 50 Hz 的控制回路来说，偶尔慢一帧比稳定慢更糟。
- **量化**：板端跑 INT8/INT16，需要真实仿真轨迹做校准采样。仓库的 README 明确写了篮球用 506 组真实观测做 max 校准并启用 INT16，球平衡和爬梯则要「移出固定归一化、拆分全连接权重为主值与残差」之后再校准。
- **定形**：输入 shape 必须固定。观测维度在不同任务上还不一样——`supervisor.py` 里 `upload_and_switch` 校验 `input_transform` 时写死 `dim = 54 if task == "ball_balance" else 61`，说明球平衡任务是 54 维观测，其余是 61 维。
- **排布与打包**：内存布局、算子合并、指令流打包在这一步确定，最后产出一个文件。

编译在 **Ubuntu 的 Docker 工具链** 里用 `hb_mapper makertbin --model-type onnx` 完成，板端不装编译器，只需要匹配版本的 HBRT/BPU runtime。这个分工是合理的：板子算力有限，而转换环境对工具链版本敏感，容器化才能保证可复现。

---

## 二、板端在环到底怎么转

`bpu_inloop_video.py` 跑起来是一条严格串起来的闭环：

```
Ubuntu：MuJoCo 物理 + EGL 渲染
    │ 每个控制步取当前观测
    ▼
TCP → RDK X5 : 8765（策略服务）
    │ BPU 载入 walking.bin，算出 14 维动作
    ◀── 动作回传
Ubuntu：动作送入仿真 → 物理步进 → 录下一帧
```

分工很干净：**BPU 只算动作，画面由工作站仿真和编码**。所以视频是「板端产生动作、工作站负责画面」的真实闭环，而不是板端录的像。

本次保存的输出里，walking 任务跑了 **500 步、重置 0 次**，BPU 单元给出 `PASS-FRESH-BPU`。GPU 侧同一任务同样 500 步，报告里 `inference_calls=500`、`real_observation_parity_checks=32`、`max_abs_output_error=4.17e-07`（atol/rtol 都是 1e-4）——也就是说 **ONNX 与 PyTorch 在同一批真实观测上的动作差在 1e-7 量级**，误差不是来自导出，而是来自后面的量化与回环。

控制步数 500、步数写死在单元里（`BPU_VIDEO_STEPS` 按任务取 1500/900/500），这种「报告里带真实控制步数」的做法很关键：一段视频只要没跑满步数，报告立刻对不上。

---

## 三、多策略：切的是什么，代价是什么

公告说「单块开发板承载多套策略，支持随时热切换」。读代码才能看清这句话的边界。

### 白名单：不是任意模型都能上板

`rdk_bpu_policy_supervisor.py` 里 `MODEL_PATHS` 预登记了 7 个任务：

```python
MODEL_PATHS = {
    "walking":      .../hbm/walking.bin,
    "perturbation": .../hbm/walking.bin,   # 与 walking 同一个模型
    "stilt":        .../hbm/stilts25.bin,
    "swing":        .../hbm/swing.bin,
    "ladder":       .../hbm/ladder.bin,
    "basketball":   .../hbm/basketball.bin,
    "ball_balance": .../hbm/ball_balance.bin,
}
```

传白名单外的名字，返回 `unknown pre-registered task`，直接拒绝。这不是限制，是保护——它保证板端永远只跑本专题登记过的模型，避免「拿 A 任务的 HBM 当 B 任务用」这种形状能对齐、行为一定错的事故。

### 切换的实现是「重启一个进程」

`switch()` 的实际动作是：

1. 若任务相同且子进程存活 → 直接返回状态（幂等，不重启）；
2. 否则 `_stop()`：terminate → 等 2 秒 → kill；
3. `Popen` 起一个新的 `rdk_bpu_policy_server.py --model <新 HBM> --port 8765`；
4. `sleep(0.5)` 后 `poll()`，进程已退出就报 `policy server exited`。

也就是说，**切换不是在一个常驻进程里换权重，而是杀掉旧服务、起新服务，让新进程重新加载 HBM 再监听同一个端口**。代价是一次进程启动 + 模型加载，切一次要几百毫秒到秒级。

这解释了为什么切换的粒度是「每个视频单元开始前切一次」，而不是「一个视频里切好几次」。想做真正的在线热切换，得改掉这套进程级调度。

### 上传有硬校验

`upload_and_switch` 做了三道检查，任何一道不过就拒绝写入：

- 体积 ≤ 16 MiB，短包直接报 `short payload`；
- `sha256` 必须和声明值一致，否则 `sha256 mismatch`；
- `input_transform` 的 offset/scale 长度必须等于 61（球平衡 54），且 scale 全部为正、全部有限。

通过后先写 `xxx.bin.uploading` 再 `replace()` 原子替换，并把 `model_sha256` 写进 `.input_transform.json`。**归一化参数和模型文件绑定在一起**——这正是 Task6 里「接口相同也不能随便换模型」在工程上的落地方式。

### 爬梯是特例：两个服务并行，不是切换

```python
PAIR_MODELS = {
    "climber": (.../climber.bin, ..._feature_scales.json, 8875),
    "getup":   (.../getup.bin,   ..._feature_combined_scales.json, 8876),
}
```

爬梯走 `{"op":"switch_pair"}`，同时起 8875 和 8876 两个服务，由工作站侧的监督器决定这一帧问谁。这是**并行双服务 + 外部仲裁**，和 walking/stilt 那种「换掉唯一服务」是完全不同的机制。

---

## 四、本次运行真的发生策略切换了吗

我的结论是：**发生了装载，没有发生跨任务切换。**

证据在保存的输出里只有一行：

```
RDK 任务策略服务已上传并切换: walking -> /home/sunrise/microduck_policy/hbm/walking.bin pid= 40560
```

这一行说明三件事：

1. 切换的目标任务就是 `walking`，本次 BPU 单元自始至终只服务这一个任务；
2. 走的是 `upload_and_switch`（带 payload），因此 `force=True`，**即使板上原本就是 walking 也会强制重启一次服务**；
3. 整段 500 步闭环里没有第二条切换记录。

所以准确的说法是：**具备切换能力，本次只做了一次启动装载，运行过程中未发生策略切换。**

顺带一个容易混淆的点：Notebook 里「切换 Notebook 时必须重新运行本单元」这句话说的「切换」，指的是**学习者换了一本 Notebook（换任务）**，不是运行时自动换策略。公告问的④，答案就在这里——如果只跑 walking，那切换只在单元开始时发生一次。

---

## 五、验收标识：这张表是这一期的核心

Notebook 第 8 节给了官方定义，我加上自己的解读：

| 标记 | 官方含义 | 我的解读 |
| --- | --- | --- |
| `PASS-local-onnx` | 模型结构、输入输出形状和有限值检查通过 | 只证明「模型能读、shape 对」，不涉及策略好坏 |
| `PASS-GPU-VIDEO` | GPU 策略连接物理仿真，完成本次视频与报告 | 本次证据，但只覆盖工作站侧 |
| `PASS-BPU-VIDEO` | 当前连接 RDK，逐步取得对应 HBM 的动作并生成视频 | **本次证据**，最强的一档 |
| `BPU-CACHED-VIDEO` | 播放之前保存的板端演示 | **历史证据**，不能证明本次板端参与 |
| `PASS-rdk-bpu-runtime-sample` | MobileNet 样例完成推理 | 只证明板子运行时活着，与任务无关 |
| `PASS-rdk-bpu` | 当前任务 HBM 的板端推理完成 | 单次推理级证据，弱于整段闭环 |

`BPU-CACHED-VIDEO` 意味着**板子离线或未通过板端校验**。它和 `PASS-BPU-VIDEO` 差的不是措辞，是证据等级：前者只能说明「仓库里存过一段板端视频」，后者才能说明「这次的每一步动作都由 BPU 算出」。

### 证据链是怎么防作伪的

这套设计最值得学的地方是**用随机编号把报告和视频绑死**：

```python
BPU_RUN_ID = str(uuid.uuid4())
bpu_env["MICRODUCK_VALIDATION_RUN_ID"] = BPU_RUN_ID
```

`run_id` 生成后注入子进程环境，子进程写进 JSON 报告，回来再用 `verify_bpu_report(video, run_id, hbm_list, board_sha_list)` 核对三件事：报告里的 `run_id` 是不是本次、模型 SHA 是不是这份 HBM、板端实际 SHA 与本地 HBM 是否一致。**少一项就对不上旧报告**，所以拿历史视频冒充本次推理在这里是过不去的。`status()` 还会回传板端模型的 sha256 和 pid，供工作站二次比对。

---

## 六、板子离线时会怎样：代码和公告说的不一样

这是本次读代码最大的发现，我打算如实写。

**公告的说法**是：教程自己遇到过板子离线、SSH 连不上，系统优雅降级——GPU 视频照常产出，BPU 环节切缓存录像，跳过板端校验，不报错终止。

**当前仓库代码**是另一回事：

```python
ALLOW_CACHED_BPU = False
...
def show_cached_bpu_video(path, reason):
    if not ALLOW_CACHED_BPU or path is None or not path.is_file():
        return False          # 开关关着，直接返回 False
    print("BPU-CACHED-VIDEO:", path)
    ...
...
else:
    if not show_cached_bpu_video(BPU_VIDEO_OUTPUT, "板端不可达，展示历史视频。"):
        raise ConnectionError("RDK 板端推理未完成；请检查开发板网络与策略服务。")
```

开关是**硬编码 False**，`show_cached_bpu_video` 因此永远返回 False，最终抛 `ConnectionError`。`03-Notebook/README.md` 里也写得很直白：「BPU 默认关闭；开启后连接失败会报错，不会自动播放旧视频冒充新推理。」

两条降级路径都要先失败才会走到缓存分支：

1. 先试 8766 控制服务（socket `timeout=8`），走 `upload_and_switch`；
2. 失败则 fallback 到 SSH（`BatchMode=yes`、`ConnectTimeout=8`），远程 `pkill` 再 `nohup` 重启策略服务；
3. 两条都失败才考虑缓存——而缓存开关是关的，于是报错。

另外，`MICRODUCK_RUN_BPU` 没开时，单元只打印 `NOT-RUN-BPU` 就结束，**连缓存视频都不放**。

我不打算替任何一边圆场。合理的推测是：**缓存分支是早期版本留下的通道，后来被显式关掉了**——因为一旦允许「连接失败 → 播旧视频」，报告里就会出现无法分辨真伪的 `BPU-CACHED-VIDEO`。把开关焊死成 False，等于在代码层面保证「要么是真推理，要么报错」。这是收紧，不是退化。

对学习者的实际意义是：**没有板子时，正确做法是让 BPU 单元报 `NOT-RUN-BPU` 或 `ConnectionError`，拿 GPU 视频交差并如实说明；不要指望系统给你自动降级出一段好看的板端视频。**

---

## 七、爬梯的两套实现，以及一处死分支

公告提醒的「爬梯有两套实现」，代码里对得很整齐：

| 方案 | 模型 | 端口 | 归属 |
| --- | --- | --- | --- |
| 社区复刻（Task6 那套） | `climber.bin` + `getup.bin` | 8875 / 8876 | `PAIR_MODELS`，官方双模型 |
| 本地 14 级 ladder | `hbm/ladder.bin` | 8765（单服务） | `MODEL_PATHS["ladder"]`，需显式 opt-in |

本地实验的门禁写在单元里：

```python
BPU_LADDER_LOCAL_OPT_IN = os.getenv("MICRODUCK_RUN_LOCAL_LADDER_BPU", "0") == "1"
...
elif BPU_TASK == "ladder" and not BPU_LADDER_LOCAL_OPT_IN:
    print("已跳过本地 14 级 ladder HBM 视频：它不是社区 climber/get-up 方案。")
    print("只有明确设置 MICRODUCK_RUN_LOCAL_LADDER_BPU=1 才会运行本地实验，不得将结果标为社区复刻。")
```

**但我判断这两行是死代码**（这是我的推断，不是仓库说明）：分支链的开头已经是

```python
if BPU_TASK == "ladder":
    ...  # 走 switch_pair，起 8875/8876
```

`BPU_TASK == "ladder"` 必定在第一个 `if` 命中，后面那个 `elif` 永远进不去。也就是说，**只要 BPU_TASK 是 ladder，跑的一定是官方 pair，本地 14 级实验在当前代码结构下根本不会被触发**。提醒文字还在，但实际门禁不在它这里。

这个发现不影响结论（本地 ladder 确实不该当复刻），但它说明一件事：**「代码里有提醒」和「代码真的会拦住」是两回事**，读这类安全提示时要顺手往上看看分支顺序。

---

## 八、没有 RDK X5，我怎么学这一期

把「跑通」换成「读懂每个会失败的环节」：

1. **读 `supervisor.py` 的白名单和 switch 实现**——弄清切换的单位是进程，不是权重；
2. **读 `bpu_inloop_video.py` 的参数**——`--bpu-host/--bpu-port` 是通信，`--hbm` 是本地校验用的，别混；
3. **核对报告字段**——`run_id`、`onnx_sha256`、板端 `sha256`、`steps`、`resets`，每一项都在回答「这段视频是谁、什么时候、跑了多久」；
4. **看 GPU 侧的 parity 数字**——`max_abs_output_error=4.17e-07` 说明导出环节几乎无损，误差要去量化和回环里找；
5. **本地用 CPU 跑同一份 ONNX** 把输入输出 shape 和 14 维动作口径对齐（关节顺序、缩放、默认姿态偏移），这一步能挡掉大多数「部署后策略变傻」。

做完这些，验收等级仍然只能停在 `NOT-RUN-BPU`。我不打算把它说成 `PASS-BPU-VIDEO`。

---

## 九、还没解决的问题

- **量化误差没有逐维数据**：INT16 相对 FP32 在 14 个关节上各差多少，我只有 GPU 侧 ONNX-vs-PyTorch 的 1e-7，缺板端 INT16 与 ONNX 的对照。
- **回环时延没有本次数字**：Task6 出现过 p95 约 167 ms 的往返，本 Notebook 的输出里没给这一项，无法判断本次 walking 的控制频率余量。
- **切换代价没实测**：进程级重启到底几百毫秒，切换瞬间动作是否跳变，都没有数据。
- **`ALLOW_CACHED_BPU` 是什么时候从 True 改成 False 的**：我没有翻提交历史，前面的「早期遗留通道」只是推测。

---

## 学习心得

Task7 把前六期串成了一条部署闭环。我理清的是三件事：ONNX 是通用交换格式，HBM 是 BPU 的编译产物（实际后缀是 `.bin`，编译在 Ubuntu Docker 里用 `hb_mapper` 完成，板端只负责加载）；板端在环是「BPU 算动作、工作站算物理和画面」，每步通过 8765 收发，8766 负责上传与切换；多策略的真相是预登记 7 个任务模型的白名单，切换的单位是**进程**——杀掉旧服务、起新服务重新加载 HBM，所以它的粒度是每个单元切一次，不是运行时热切换。本次保存的输出里只有一条 `upload_and_switch: walking`，因此准确结论是「具备切换能力，本次只做了一次启动装载，运行中未发生跨任务切换」。

验收上我记住了 `PASS-BPU-VIDEO` 与 `BPU-CACHED-VIDEO` 的区别：前者是本次证据，后者是历史录像，标 CACHED 就等于说板子当时不在线。防作伪靠的是随机 `run_id` 注入子进程、再与模型 SHA 和板端 SHA 三方核对。

还有两处我特意没有替公告圆场：当前仓库里 `ALLOW_CACHED_BPU = False` 是硬编码，缓存分支走不到，板端连不上会直接 `ConnectionError`，README 也写明「不会自动播放旧视频冒充新推理」——这和公告描述的「优雅降级」不一致，我倾向于认为缓存通道是早期遗留、后来被收紧了。另外爬梯本地 14 级实验的那两行 opt-in 提示，因为分支顺序问题实际是死代码，只要 `BPU_TASK == "ladder"` 就一定走官方 pair。

---

## 课程建议

1. **把 `ALLOW_CACHED_BPU` 的口径讲清楚**。公告的「优雅降级」和仓库的「连接失败会报错」是两种说法，学习者很容易照着公告期待一段缓存视频，结果撞上 `ConnectionError`。建议在 Notebook 里补一句版本说明：缓存通道何时存在、何时关闭、为什么关闭。
2. **给一张板端探测的分支图**。现在是 socket(8s) → SSH(8s) → 报错三层，读代码能看懂，但第一次跑的人只会看到最后一行异常。画出来比读脚本快得多。
3. **回环时延应该进报告摘要**。Task6 的 p95 167 ms 是直接能解释「切换条件凑不齐」的关键量，这一期却在报告里找不到它。时延是任务机制的一部分，不该只在别处出现。
4. **`--hbm` 参数命名容易误导**。它其实是工作站本地的编译产物路径，只用于算 SHA 和上传。建议改名成 `--local-hbm-for-upload` 之类，或在参数说明里写清「不是板端路径」。
5. **修一下 ladder opt-in 的死分支**，或者把提醒挪到真正生效的位置。安全提示放在永远不执行的分支里，比没有提示更危险——它会让人以为门已经关好了。
6. **给一份不依赖实体板的最小验证路径**：mock 一个 8765 服务按 ONNX 返回动作，让通信、超时、报告核对和失败分支都能被跑一遍。没有硬件的人现在只能读代码，无法验证自己的理解对不对。

---

## 个人学习评级

- ONNX / HBM / BPU 三者的分工与编译边界：A-
- 板端在环的数据流与端口职责：A
- 多策略白名单与进程级切换机制：A
- 验收标识与 run_id / SHA 证据链：A
- 降级路径与公告口径差异的辨析：B+
- 本次综合掌握度：A-

---

## 课程资料

- [Notebook 07：RDK 网页与多策略 BPU 验收](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/05-MicroDuck%E8%BF%90%E5%8A%A8%E7%AD%96%E7%95%A5%E4%B8%8E%E4%BB%BB%E5%8A%A1%E5%BC%80%E5%8F%91%E4%B8%93%E9%A2%98/03-Notebook/07_RDK%E7%BD%91%E9%A1%B5%E4%B8%8E%E5%A4%9A%E7%AD%96%E7%95%A5_BPU%E9%AA%8C%E6%94%B6.ipynb)
- [03-Notebook 总览（本轮验证、GPU/BPU 视频、模型与 BPU 注入）](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/05-MicroDuck%E8%BF%90%E5%8A%A8%E7%AD%96%E7%95%A5%E4%B8%8E%E4%BB%BB%E5%8A%A1%E5%BC%80%E5%8F%91%E4%B8%93%E9%A2%98/03-Notebook/README.md)
- 板端调度源码：`02-可运行代码/microduck-playground-stilts/scripts/rdk_bpu_policy_supervisor.py`
- 本轮报告：`recordings/gpu_videos/walking_gpu_latest.json`、`recordings/bpu_videos/walking_bpu_latest.json`（`run_id=92aa7fcd-c496-4c0f-a43f-28f4abb39b45`，`generated_at_utc=2026-10-11T08:57:59Z`）
