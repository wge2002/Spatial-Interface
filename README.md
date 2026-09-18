# Spatial Interface

通过视觉、局部几何参照和短动作程序，让通用模型操作仿真机器人。
当前默认接口是 **direct geometry + grounded feedback**：模型选择 RGB-D
图像区域、构造动作，并读取与实际末端运动对应的视觉和数值反馈。

仓库：[`wge2002/Spatial-Interface`](https://github.com/wge2002/Spatial-Interface)。
Python 包名是 `spatial_interface`，依赖项目名是 `spatial-interface`。
首版从已有 DG-r5 实现迁入，使用新的 `si-r1` 身份和独立 Git 历史；
来源及迁移范围见 [ATTRIBUTION.md](ATTRIBUTION.md)。

## 1. 克隆与环境

需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、Python 3.10
（uv 可自动安装），以及 PATH 中带 `libx264` 编码器的 FFmpeg，用于保存相机视频。
Linux 仿真建议使用有 NVIDIA 驱动和 EGL 的机器。
完整视频、环境和运行日志保存在被 Git 忽略的 `data/` 下。

```bash
git clone --recurse-submodules git@github.com:wge2002/Spatial-Interface.git
cd Spatial-Interface
bash scripts/bootstrap.sh
source set_env.sh
```

`bootstrap.sh` 按 `uv.lock` 安装独立 `.venv`，安装 Playwright Chromium，
并生成本项目的 `.libero/config.yaml`。LIBERO 子模块固定在
`8f1084e3132a39270c3a13ebe37270a43ece2a01`。不会修改 `~/.libero`。
安装后仍须 `source set_env.sh`；直接运行 `uv run` 不会替代这里的环境配置。

无桌面的 Ubuntu/Debian x86_64 服务器使用：

```bash
bash scripts/bootstrap.sh --headless
source set_env.sh
export MUJOCO_EGL_DEVICE_ID=0  # 按 nvidia-smi 选择空闲、当前进程可见的 GPU
source scripts/headless_env.sh
```

可选 headless 安装把缺失的 Chromium 库解压到 `data/env/sysroot`，不使用 sudo。
该容器配置使用 Chromium `--no-sandbox`，仅用于运行本项目本地仿真页面。
如果 GitHub HTTPS 子模块下载失败而 SSH 可用，可在本仓库设置
`git config submodule.third_party/LIBERO.url git@github.com:Lifelong-Robot-Learning/LIBERO.git`
后重试安装。

## 2. 原生 Codex / Astra 配置

默认条件：**JKWL `/v1` Responses，`gpt-6-astra`，`medium`，原生 Codex 0.153.4**。
机器人 MCP 必须初始化成功后才开始推理。配置通过项目私有 `CODEX_HOME` 加载，
没有自动切换供应商的路径。有关配置项，见 [官方 MCP 文档](https://developers.openai.com/zh-Hans/docs/extend/mcp)。

已安装该版本 Codex 时：

```bash
python scripts/setup_codex.py --binary /absolute/path/to/codex
source set_env.sh
```

也可以使用已有 Node.js/npm，安装在本项目内：

```bash
npm install --prefix data/env/codex-npm @openai/codex@0.153.4
python scripts/setup_codex.py --binary data/env/codex-npm/node_modules/.bin/codex
source set_env.sh
```

`setup_codex.py` 会检查客户端版本，仅写入不含凭据的配置。
已有安装必须保留完整运行目录。需要复制原生安装时，在命令后加 `--copy-native`；
脚本会一并复制 `codex-code-mode-host`、`codex-resources` 和 `rg`，不能只复制 `codex`。

### 手动填写并加载 API key

仓库不包含真实密钥。首次配置时，在**运行实验那台机器的项目根目录**
新建本地文件 `.env.local`，例如用编辑器打开：

```bash
nano .env.local
```

粘贴下面这一行，将单引号内的占位文字替换为自己的 JKWL 密钥并保存：

```bash
export SPATIAL_JKWL_API_KEY='在这里粘贴你的JKWL密钥'
```

首次保存后，将文件权限设为仅当前用户可读写：

```bash
chmod 600 .env.local
```

**每次打开新终端，进入项目根目录后，在启动实验的同一个终端执行：**

```bash
source set_env.sh
source .env.local
```

无桌面的 Linux 服务器再执行 `source scripts/headless_env.sh`。
`.env.local` 已被 `.gitignore` 忽略，正常 Git 提交不会包含它；
目前不会自动加载此文件，因此需要手动 `source .env.local`。
不要把真实密钥填入已跟踪的 `config/codex-jkwl.toml` 或 `set_env.sh`。
已有安全凭据工具时，也可直接向当前终端导出 `SPATIAL_JKWL_API_KEY`。

## 3. 查看任务和启动实时可视化

```bash
python -m spatial_interface.experiment --list-tasks

# 手动查看仿真，不调用模型；Ctrl-C 结束并清理本次启动的进程。
SPHINX_BASE_PORT=8300 python -m spatial_interface.record_sim \
  --task stack --seed 1 --render 0 --data_root data/manual
```

桌面机器打开 [http://localhost:8300](http://localhost:8300)。
`--render 0` 关闭 MuJoCo 原生窗口，浏览器三维界面仍然运行。
浏览器可显示点云、相机图像与夹爪；这与模型通过 MCP 收到的观察属于同一仿真。

在远端启动时，另开一个**本机终端**建立转发；把 `your-gpu-host` 换成自己的 SSH 主机：

```bash
ssh -N \
  -L 8300:127.0.0.1:8300 \
  -L 8301:127.0.0.1:8301 \
  -L 8302:127.0.0.1:8302 \
  your-gpu-host
```

然后在本机打开 `http://localhost:8300`。必须同时转发这三个端口；
本机和远端的端口号码应一致。每个实例占用四个连续端口：

| 相对基准端口 | 用途 | 远程查看是否需要转发 |
| --- | --- | --- |
| +0 | 浏览器页面、模型资源 | 是 |
| +1 | 相机、点云 WebSocket 和 JSON | 是 |
| +2 | 页面控制 WebSocket | 是 |
| +3 | 远端 Chromium CDP，供机器人 MCP 连接 | 通常不需要 |

端口已被占用时程序直接退出，不会终止已有服务。多个实例选择不重叠的四端口组。
手动仿真与下面的自动实验各自启动仿真；示例分别使用 8300 和 8400。

不配置模型也可验证仿真、浏览器和真实 MCP 图像返回：

```bash
python scripts/smoke_mcp.py --task stack --seed 31 --base-port 8500 \
  --out data/smoke/mcp_001
```

它只调用观察和结束工具，保存预览图与 `verification.json`，结束后清理自己的进程。

## 4. 启动一次新实验

先完成环境和模型配置，然后预览计划：

```bash
python -m spatial_interface.experiment \
  --task stack --seed 1 --base-port 8400 \
  --out data/runs/stack_seed1_001 --dry-run
```

确认打印的条件符合预期后，去掉 `--dry-run` 执行：

```bash
python -m spatial_interface.experiment \
  --task stack --seed 1 --base-port 8400 \
  --out data/runs/stack_seed1_001
```

默认单回合模型运行上限为 1800 秒，环境初始化和退出另计；无自动重试。
可用 `--timeout` 调整，`--model` / `--effort` 显式改变模型条件。
已存在的 `--out` 一律拒绝覆盖；每次新尝试都用新目录。
在远端运行时，使用上一节的 SSH 命令，将三个端口改为 **8400/8401/8402**，
打开 `http://localhost:8400` 即可实时查看这次实验。

七个默认任务是 `stack`、`t_block`、`rainbow`、`libero_goal/0`、
`libero_goal/4`、`libero_goal/7`、`libero_goal/8`。Rainbow 需要人工评分；
不要把原始自动字段当成它的正式成功率。

只验证连接、不求解任务时，可运行观察/结束冒烟测试：

```bash
python -m spatial_interface.experiment --smoke \
  --task stack --seed 31 --base-port 8500 \
  --out data/smoke/native_001
```

该模式仍调用模型，但要求只做一次 `dg_look` 然后 `end_episode`，不提交运动程序；
模型运行上限不超过 180 秒，结果标记为 `smoke-not-scored`。

## 5. 结果与视频回放

每次输出包含 `manifest.json`、`method_identity.json`、`result.json`、日志，
以及 `episodes/<task>/seed<N>/` 下的原始 `verdict.json` 和客户端记录。
有运动帧时会保存相机视频；仅观察后结束的冒烟测试没有运动帧，因此通常不产生视频。
方法身份记录实际完整 Git 提交、发布文件哈希、客户端档案与 LIBERO 版本；
有未提交的方法改动时拒绝正式启动。超时记录与官方 verdict 分开保存。

```bash
python -m spatial_interface.build_video_index data/runs/stack_seed1_001/episodes
python -m http.server 8600 --bind 127.0.0.1 \
  --directory data/runs/stack_seed1_001/episodes
```

打开 [http://localhost:8600/videos.html](http://localhost:8600/videos.html)。
远程回放另建 `ssh -N -L 8600:127.0.0.1:8600 your-gpu-host`；回放只需要一个端口。
`cameras.mp4` 是相机录像。视频省略等待所产生的时间差不能当作完整任务耗时。

## 开发与方法说明

```bash
python -m pytest -q tests
python scripts/methods.py validate
python scripts/methods.py snapshot --out data/identity/check.json
```

- [当前直接几何方法指南](docs/DIRECT_GEOMETRY_GUIDE.md)
- [方法身份与迁移规则](methods/README.md)
- [验证范围](docs/VALIDATION.md)

内部 `VIA_*` 和 `SPHINX_BASE_PORT` 环境变量及 `sphinx2` MCP 标识保留兼容性。
对外仓库、安装项目、Python 命令和页面标题均使用新名称。
旧接口仍在代码中，使用 `--interface` 显式选择；当前默认是 direct_geometry。
Qwen 客户端保留为可选入口（`--harness qwen --model <server-model> --effort low`），
需要自己的 `VIA_QWEN_SERVER_CONFIG`，不在本次原生 Astra 启动验收范围内。
