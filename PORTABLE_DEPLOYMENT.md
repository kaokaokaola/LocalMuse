# LocalMuse V2 — 便携化 Python 运行环境

## 1. 这是什么 / 解决什么问题

目前 `LocalMuse.bat` 依赖 `.\venv\`（Python 虚拟环境）运行。普通 venv 在
`venv\pyvenv.cfg` 里记录了创建它时所用的「基础 Python」安装路径
（例如 `home = C:\Program Files\Python310`）。把 `venv\` 文件夹直接复制到
另一台电脑后，`venv\Scripts\python.exe` 仍会去找这个原电脑上的路径，
找不到就无法运行——这就是 venv「不能直接复制」的根本原因。

`build_portable_python.bat` 会在项目根目录生成一个新的 `python\` 文件夹，
把：

1. 当前 venv 里**已经装好的全部依赖**（FastAPI、PyTorch CUDA、
   transformers、flash-attn、faiss、easyocr 等，约 6GB）
2. 加上一份**独立的 Python 解释器**（标准库 + DLL，从 venv 所引用的
   基础 Python 安装中复制）

合并打包到 `python\` 里，生成一个**完全自包含、免安装**的 Python 运行环境。

`LocalMuse.bat` 本身已经写好了优先级逻辑：

```
venv\Scripts\python.exe  >  python\python.exe  >  系统 PATH 中的 python
```

所以**不需要改任何代码** —— 只要 `python\` 文件夹存在，`LocalMuse.bat`
在找不到 `venv\` 时就会自动用它。

## 2. 如何生成

1. 确保 `.\venv\` 是好的（即 `LocalMuse.bat` 目前能正常运行）。
2. 关闭正在运行的 LocalMuse（避免文件被占用）。
3. 确认磁盘至少有 **7GB 可用空间**（会生成一份新的 ~6GB 拷贝，
   不会删除原 venv）。
4. 双击运行 `build_portable_python.bat`。
   - 第 1 步：复制基础 Python 解释器 + 标准库（几十 MB，很快）。
   - 第 2 步：复制 `venv\Lib\site-packages`（~6GB，**这一步最慢**，
     几分钟到十几分钟，取决于硬盘速度）。
   - 第 3 步：用新生成的 `python\python.exe` 单独验证
     `torch` / `fastapi` / `faiss` 等能否正常导入，并打印
     `CUDA available: True/False`。

完成后项目结构大致是：

```
2026.05.24.LocalMuse.V2\
├─ LocalMuse.bat
├─ main.py
├─ src\
├─ venv\        <- 原虚拟环境（保留作为本机的运行环境）
└─ python\      <- 新生成的便携式运行环境（~6GB）
```

## 3. 怎么迁移到另一台电脑

把**整个项目文件夹**复制过去，但**不要复制 `venv\`**
（不需要，`python\` 已经包含等价内容，省下 6GB 传输量）。

可以排除/不复制的内容：

- `venv\`（不需要）
- `__pycache__\`、`localmuse_crash.log`（运行时产物，会自动重新生成）

到新电脑后，直接双击 `LocalMuse.bat`：因为没有 `venv\`，它会自动
fallback 到 `python\python.exe`，**无需安装 Python、无需 pip
install、无需运行 setup.bat**。

打开 LocalMuse 后，在「打开图库」里重新选择图像库所在的文件夹即可
（图像库本身存放在你指定的目录，不在项目文件夹里，所以不会随项目
打包，需要你在新电脑上能访问到同一个图库目录，比如外接硬盘或
网络共享）。

## 4. 重要前提条件（新电脑需要满足）

### 4.1 操作系统 / 架构
新电脑必须是 **Windows x64**，且 CPU 架构、Windows 版本与生成
`python\` 时使用的基础 Python（当前是 Python 3.10.1 / `cp310`
ABI）兼容。一般来说现代 64 位 Windows 10/11 都没问题。

### 4.2 显卡驱动（GPU 加速）
当前 `python\Lib\site-packages\torch` 是 **PyTorch 2.6.0+cu124**，
里面已经**内置了 CUDA 12.4 运行时所需的 DLL**
（`cudart64_12.dll`、`cublas64_12.dll`、`cudnn64_9.dll` 等），
**不需要**在新电脑上单独安装 CUDA Toolkit。

但**显卡驱动（Driver）是 Windows 系统级组件，无法打包进文件夹**，
新电脑需要：

- 一块 **NVIDIA 显卡**
- 安装了**支持 CUDA 12.4 的驱动**（建议 2024 年中以后发布的
  Game Ready / Studio 驱动；可在新电脑上用 `nvidia-smi` 查看驱动
  支持的最高 CUDA 版本，应 ≥ 12.4）

如果新电脑**没有 NVIDIA 显卡 / 驱动版本太旧**，PyTorch 会自动
fallback 到 CPU 模式（`torch.cuda.is_available()` 返回 `False`），
LocalMuse 仍可运行（语义检索/UI 都正常），但 **VLM 标注
（Annotate）和深度/姿态相关推理会变得非常慢**，不建议在 CPU 上跑
大批量标注。

### 4.3 模型缓存（首次运行可能需要重新下载）
以下内容**不在 `python\` 或 `venv\` 里**，而是缓存在系统用户目录下
（例如 `C:\Users\<用户名>\.cache\huggingface\`、
`C:\Users\<用户名>\.cache\torch\`、`C:\Users\<用户名>\.EasyOCR\`
等），属于「首次使用时自动下载」：

- CLIP 语义模型权重
- Qwen3-VL / Qwen2.5-VL 标注模型权重（7B 模型约 15-16GB）
- EasyOCR 检测/识别模型
- 深度估计 / 姿态估计模型权重

如果新电脑也能联网，这些会在第一次用到对应功能时自动下载，
**不影响搜索等核心功能的迁移**，只是第一次用「Annotate」
（标注）功能时会先等待下载。

如果新电脑**不方便联网下载**（例如内网环境），可以把上述
`.cache` 目录里对应的子文件夹也一并拷贝到新电脑的同一路径下，
这样就不需要重新下载了（可选，体积较大，按需拷贝）。

## 5. 之后还需要装新的 Python 包怎么办？

`python\` 里已经包含了完整的 `pip`，可以这样用（不需要
`pip.exe`，直接用 `-m pip`）：

```bat
python\python.exe -m pip install <package_name>
```

## 6. 常见问题

**Q: 生成 `python\` 之后，`venv\` 还能删吗？**
A: 可以。确认 `LocalMuse.bat` 能在没有 `venv\` 的情况下正常启动
（即只剩 `python\` 时也能跑）后，删除 `venv\` 可以在本机也省下
约 6GB 空间。以后用 `setup.bat` / `setup_annotation.bat` /
`install_flash_attn.bat` 重新升级依赖时，记得针对 `python\` 而不是
`venv\`（或者重新跑一遍 `setup.bat` 重建 `venv\`，再重新执行一次
`build_portable_python.bat` 同步到 `python\`）。

**Q: 复制 6GB 的 `python\` 文件夹时报错 / 路径太长？**
A: Windows 资源管理器自带的 ZIP 压缩/解压在处理深层嵌套路径时
（PyTorch 的 `include\` 目录层级很深）有时会报「路径过长」。
建议：
- 直接用文件资源管理器**整个文件夹拖拽复制**（不压缩），或
- 用 **7-Zip** 压缩/解压（对长路径支持更好），或
- 用 `robocopy` 通过网络共享直接同步文件夹。

**Q: 想进一步缩小体积？**
A: `python\Lib\site-packages\torch\lib\` 里的 CUDA DLL（约 4.3GB）
是体积大头，是 GPU 推理必需的，不建议删除。如果确定目标机器
**只用 CPU**，理论上可以把 venv 换成 CPU 版 PyTorch
（`pip install torch --index-url https://download.pytorch.org/whl/cpu`，
约几百 MB）后再运行本脚本，体积会显著缩小，但会失去 GPU 加速。
