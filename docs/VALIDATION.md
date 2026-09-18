# 验证范围

2026-09-18 在 Linux x86_64 / NVIDIA EGL 上验证了独立安装和实际启动。
项目使用自己的 Python 3.10.20 `.venv`、LIBERO 配置、Chromium 库目录、
Codex 运行组件和 `CODEX_HOME`。凭据仅通过已有安全环境认证导入。

| 检查 | 结果 |
| --- | --- |
| `bash scripts/bootstrap.sh --headless` | 通过；按锁文件检查依赖，生成本项目 LIBERO 配置和 headless 环境 |
| `python -m pytest -q tests` | **459 passed**；包含端口占用保护、方法身份、完整 Codex 运行组件复制、视频编码和导入的核心测试 |
| 公开 CLI | 七个任务列表、默认模型/接口、dry-run、新输出目录要求均已检查 |
| `python scripts/methods.py validate` | 通过；核对代码哈希、唯一根提交和 LIBERO 固定版本 |
| 真实仿真 / 浏览器 / MCP | `stack` 和 `libero_goal/8`，seed 31；各返回两张真实观察图并正常结束，未提交运动程序 |
| 原生 Astra | Codex 0.153.4，JKWL Responses，`gpt-6-astra` / `medium`；`dg_look` 一次返回两张图，随后 `end_episode` 成功；零运动步 |
| 视频回放 | 12 帧合成播放样例经实际录像编码器生成 H.264；README 中的索引和 HTTP 服务命令可用，页面/视频均 HTTP 200，浏览器成功加载 640×480 视频 |
| macOS | Python 语法、bash/zsh 语法及 11 项公开入口/运行组件回归检查通过；未在 macOS 验收完整仿真依赖安装 |

原生测试的 `attempt_status` 为 `completed`，结果标记为 `smoke-not-scored`。
只观察后结束没有尝试堆叠，环境的 `success: false` 不代表连接检查失败。
此类零运动测试通常没有相机视频；回放检查使用明确标记的合成帧样例，
不作为机器人实验结果。

迁移时对源代码进行命名空间归一化比较：26 个导入模块的 Python AST
与来源一致；`record_sim.py` 和 `run_eval.py` 的额外变化是端口占用时拒绝
启动，避免终止已有服务。直接几何指南保持原文，`CLAUDE.md` 仅更新包路径。
新增的是独立环境、启动入口与方法身份记录。

原生验收发现并修复了单独复制 `codex` 会遗漏工具运行组件的问题。
`--copy-native` 现在校验并复制 `codex-code-mode-host`、`codex-resources`
和 `rg`，缺少组件会在安装阶段报错。

原生检查完成后仅更新 README 和验证说明；最终发布时重新校验方法文件哈希。
完整运行记录保留在被 Git 忽略的输出目录中。未进行完整基准矩阵、训练、
物理机器人操作或 Qwen 在线验收，也不把旧 VIA 的评测分数当作本仓库成绩。
