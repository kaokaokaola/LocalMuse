

https://github.com/user-attachments/assets/6c7b313f-46c2-472c-9c5f-7335e6ee6a13


# LocalMuse V2.1 使用说明 / User Guide

LocalMuse 是一个本地运行的图片资料库和灵感检索工具，面向建筑、空间、室内、景观、产品参考图等视觉资料管理场景。它可以在本机建立 `.library` 图库，导入图片、生成缩略图和向量索引，并支持语义搜索、草图搜索、颜色搜索、重复/翻转图像检测、软删除回收站和自动标签。

LocalMuse is a local image library and inspiration search tool for architecture, spatial design, interiors, landscape, product references, and similar visual collections. It creates a `.library` database on your machine, imports images, builds thumbnails and vector indexes, and supports semantic search, sketch search, color search, duplicate/flipped-image detection, soft-delete trash, and automatic tagging.

## 目录 / Contents

1. [部署环境 / Deployment Environment](#1-部署环境--deployment-environment)
2. [安装 / Installation](#2-安装--installation)
3. [启动 / Launch](#3-启动--launch)
4. [第一次建库 / First Library](#4-第一次建库--first-library)
5. [导入图片 / Import Images](#5-导入图片--import-images)
6. [搜索与浏览 / Search and Browse](#6-搜索与浏览--search-and-browse)
7. [图片管理 / Image Management](#7-图片管理--image-management)
8. [重复与翻转图像检测 / Duplicate and Flipped Image Detection](#8-重复与翻转图像检测--duplicate-and-flipped-image-detection)
9. [自动标签 / Automatic Tagging](#9-自动标签--automatic-tagging)
10. [图库维护 / Library Maintenance](#10-图库维护--library-maintenance)
11. [数据结构 / Data Layout](#11-数据结构--data-layout)
12. [常见问题 / Troubleshooting](#12-常见问题--troubleshooting)
13. [隐私说明 / Privacy](#13-隐私说明--privacy)

## 1. 部署环境 / Deployment Environment

### 中文

推荐环境：

- 操作系统：Windows 10 或 Windows 11。
- Python：建议 Python 3.10、3.11 或 3.12。
- Git：建议安装。`setup.bat` 会通过 Git 安装 OpenAI CLIP 依赖；如果没有 Git，语义搜索相关能力可能无法完整安装。
- 浏览器：Chrome、Edge、Firefox 等现代浏览器均可。
- 硬件：CPU 可运行。图库较大时，建议准备较充足的内存和磁盘空间。
- 网络：首次安装和首次加载模型时可能需要下载依赖或模型文件。

LocalMuse 默认在本机运行，不需要部署到云服务器。所有图库文件、索引、缩略图和标签都保存在本地 `.library` 文件夹中。

### English

Recommended environment:

- OS: Windows 10 or Windows 11.
- Python: Python 3.10, 3.11, or 3.12 is recommended.
- Git: recommended. `setup.bat` uses Git to install the OpenAI CLIP dependency. Without Git, semantic-search features may not be installed completely.
- Browser: any modern browser such as Chrome, Edge, or Firefox.
- Hardware: CPU is supported. For large libraries, use enough memory and disk space.
- Network: the first installation and first model load may download dependencies or model files.

LocalMuse runs locally by default. There is no cloud deployment requirement. Library data, indexes, thumbnails, and tags are stored inside local `.library` folders.

## 2. 安装 / Installation

### 中文

在项目根目录双击运行：

```bat
setup.bat
```

脚本会自动完成以下工作：

- 创建或复用 `venv` 虚拟环境。
- 安装 CPU 版本 PyTorch。
- 安装 FastAPI、Uvicorn、Pillow、OpenCV、NumPy、FAISS 等基础依赖。
- 安装 OpenAI CLIP。
- 尝试安装可选增强依赖，例如 YOLO、EasyOCR、M-CLIP。

如果安装过程中出现网络中断，可以重新运行 `setup.bat`。脚本会复用已有虚拟环境并继续安装缺失依赖。

如果提示找不到 Git，请先安装 Git for Windows，然后重新运行：

```bat
setup.bat
```

### English

From the project root, double-click or run:

```bat
setup.bat
```

The script will:

- Create or reuse the `venv` virtual environment.
- Install the CPU build of PyTorch.
- Install base dependencies such as FastAPI, Uvicorn, Pillow, OpenCV, NumPy, and FAISS.
- Install OpenAI CLIP.
- Try to install optional enhancement packages such as YOLO, EasyOCR, and M-CLIP.

If installation is interrupted by a network issue, run `setup.bat` again. The script will reuse the existing virtual environment and continue installing missing packages.

If Git is missing, install Git for Windows first, then run:

```bat
setup.bat
```

## 3. 启动 / Launch

### 中文

推荐方式：双击运行：

```bat
LocalMuse.bat
```

启动后程序会打开本地服务，并自动访问：

```text
http://127.0.0.1:17788
```

也可以用命令行启动：

```bat
venv\Scripts\python.exe main.py --port 17788
```

如果不希望自动打开浏览器：

```bat
venv\Scripts\python.exe main.py --no-browser
```

停止程序：在启动窗口中按 `Ctrl+C`。

如果程序异常退出，查看项目根目录下的：

```text
localmuse_crash.log
```

### English

Recommended launch method: double-click:

```bat
LocalMuse.bat
```

The app starts a local service and opens:

```text
http://127.0.0.1:17788
```

You can also launch it from the command line:

```bat
venv\Scripts\python.exe main.py --port 17788
```

To start without opening a browser automatically:

```bat
venv\Scripts\python.exe main.py --no-browser
```

To stop the app, press `Ctrl+C` in the launch window.

If the app exits unexpectedly, check:

```text
localmuse_crash.log
```

## 4. 第一次建库 / First Library

### 中文

启动后，先创建或打开一个图库。

创建新图库：

1. 点击新建图库。
2. 选择图库保存位置。
3. 输入图库名称。
4. LocalMuse 会创建一个以 `.library` 结尾的图库文件夹。

打开已有图库：

1. 点击打开图库。
2. 选择已有的 `.library` 文件夹。
3. 程序会加载已有图片、缩略图、索引和标签。

图库是 LocalMuse 的核心工作区。不同项目可以使用不同 `.library`，例如：

```text
Architecture_References.library
Interior_Moodboard.library
Material_Studies.library
```

### English

After launch, create or open a library first.

To create a new library:

1. Click the create-library action.
2. Choose where to save the library.
3. Enter a library name.
4. LocalMuse creates a folder ending with `.library`.

To open an existing library:

1. Click the open-library action.
2. Select an existing `.library` folder.
3. The app loads existing images, thumbnails, indexes, and tags.

A library is the main LocalMuse workspace. Different projects can use different `.library` folders, for example:

```text
Architecture_References.library
Interior_Moodboard.library
Material_Studies.library
```

## 5. 导入图片 / Import Images

### 中文

支持两种导入方式：

- 导入文件：选择一张或多张图片。
- 导入文件夹：递归扫描文件夹中的图片。

导入时，LocalMuse 会生成：

- 图片记录。
- 缩略图。
- 文件哈希。
- 感知哈希。
- 语义向量索引。
- 草图/轮廓等辅助索引。
- 可用于后续管理的标签数据。

支持的常见格式包括：

```text
jpg, jpeg, png, bmp, webp
```

导入大量图片时请保持程序窗口打开。首次导入会比后续操作更慢，因为模型需要加载，图片也需要建立索引。

### English

There are two import methods:

- Import files: select one or more images.
- Import folder: recursively scan a folder for images.

During import, LocalMuse creates:

- Image records.
- Thumbnails.
- File hashes.
- Perceptual hashes.
- Semantic vector indexes.
- Sketch/edge helper indexes.
- Tag data for later management.

Common supported formats include:

```text
jpg, jpeg, png, bmp, webp
```

Keep the app window open when importing many images. The first import can be slower because models must load and images must be indexed.

## 6. 搜索与浏览 / Search and Browse

### 中文

主界面可以浏览当前图库中的图片。点击图片卡片可以查看详情。

常用搜索方式：

- 语义搜索：输入自然语言，例如 `concrete courtyard with soft light`、`木结构室内空间`。
- 草图搜索：使用草图或轮廓信息匹配相似构图。
- 颜色搜索：按颜色倾向查找图片。
- 实时搜索：开启 Live 后，输入内容变化时自动触发搜索。

建议：

- 英文语义搜索通常更稳定。
- 中文也可以尝试，具体效果取决于本地已安装的模型能力。
- 搜索结果不好时，可以换成更具体的视觉描述，例如材质、空间类型、光线、构图、颜色。

### English

The main screen browses images in the current library. Click an image card to inspect it.

Common search modes:

- Semantic search: enter natural language, such as `concrete courtyard with soft light` or `wooden interior space`.
- Sketch search: match similar composition using sketch or edge information.
- Color search: find images by color tendency.
- Live search: when Live is enabled, search runs automatically as the query changes.

Tips:

- English semantic queries are usually more stable.
- Chinese queries can also be used, depending on the locally installed model capability.
- If results are weak, use more specific visual descriptions such as material, space type, lighting, composition, and color.

## 7. 图片管理 / Image Management

### 中文

在图片卡片或详情面板中，可以进行常见管理操作：

- 查看大图。
- 打开原图所在位置。
- 编辑标签。
- 从当前图库中删除图片记录。

普通图片删除会从 LocalMuse 图库和搜索索引中移除该图片。默认情况下不会删除原始源文件，除非接口或功能明确启用了删除源文件。

重复图像面板中的删除使用软删除机制。软删除会把图片移入当前图库的 `.trash` 回收区，并支持撤销或恢复。

### English

From an image card or the detail panel, you can:

- View the image.
- Open the original file location.
- Edit tags.
- Remove the image from the current library.

Normal image deletion removes the image from the LocalMuse library and search indexes. By default, the original source file is preserved unless source-file deletion is explicitly enabled by the operation.

Deletion inside the duplicate panel uses soft delete. Soft-deleted images are moved into the current library's `.trash` area and can be undone or restored.

## 8. 重复与翻转图像检测 / Duplicate and Flipped Image Detection

### 中文

LocalMuse 支持检测：

- 完全重复图像。
- 高相似图像。
- 水平翻转后的重复图像。

使用方式：

1. 打开图库后，点击 `Duplicates`。
2. 调整相似度阈值。默认值通常适合先做保守检查。
3. 点击扫描。
4. 在结果组中查看相似图片。
5. 使用对比视图查看两张图的差异。
6. 勾选需要删除的版本。
7. 点击删除所选项。

常用策略：

- `Keep Largest`：每组保留分辨率或文件体积最大的版本，其他版本标记为待删除。
- `Delete Selected`：软删除已选择图片。
- `Also delete source`：同时删除源文件。此操作风险更高，使用前请确认。
- `Undo`：误删后可在提示条中撤销。
- `Purge Trash`：清空回收区，永久删除回收区中的文件。

建议先只执行软删除，不要立刻清空回收区。确认图库状态无误后，再清理回收区。

### English

LocalMuse can detect:

- Exact duplicates.
- Highly similar images.
- Horizontally flipped duplicates.

Workflow:

1. Open a library and click `Duplicates`.
2. Adjust the similarity threshold. The default is usually a conservative starting point.
3. Start the scan.
4. Review similar-image groups.
5. Use the compare view to inspect differences between two images.
6. Select versions to delete.
7. Click delete selected.

Common actions:

- `Keep Largest`: keep the largest or highest-resolution image in each group and mark the rest for deletion.
- `Delete Selected`: soft-delete selected images.
- `Also delete source`: delete source files too. Use this carefully.
- `Undo`: restore images immediately after accidental deletion.
- `Purge Trash`: permanently remove files from the trash.

It is safer to soft-delete first and avoid purging the trash immediately. Purge only after confirming the library looks correct.

## 9. 自动标签 / Automatic Tagging

### 中文

点击 `Auto Tag` 可以为图库图片自动生成标签。

自动标签主要用于建筑和设计图片的快速整理，会根据图像内容匹配常见空间、材料、风格、构件等词汇。标签数量会被控制，避免每张图产生过多噪音标签。

特点：

- 使用已有图像向量，不重复编码图片。
- 可批量处理当前图库。
- 标签会写入图库数据，后续可用于筛选和管理。

自动标签是辅助工具，不是最终分类标准。建议在重要项目中人工复核关键图片标签。

### English

Click `Auto Tag` to generate tags for library images automatically.

Automatic tagging is designed for architecture and design references. It matches images against common terms for spaces, materials, styles, components, and related visual attributes. The number of tags is limited to avoid noisy over-tagging.

Features:

- Reuses existing image vectors instead of re-encoding images.
- Can process the current library in batch.
- Writes tags into the library data for later filtering and management.

Automatic tags are an assistant, not a final classification standard. For important projects, review key image tags manually.

## 10. 图库维护 / Library Maintenance

### 中文

常用维护功能：

- 检查图库状态：确认当前图库是否正常。
- 补全数据：为旧图库补充缺失的哈希、索引或元数据。
- 重复扫描：查找重复、相似、翻转图片。
- 自动标签：批量补充标签。
- 回收区清理：确认无误后永久清理软删除文件。

建议维护顺序：

1. 打开图库。
2. 执行图库检查。
3. 如提示缺失数据，先执行补全。
4. 扫描重复图片。
5. 软删除确认无用的重复项。
6. 运行自动标签。
7. 确认无误后再清理回收区。

### English

Common maintenance actions:

- Check library status: verify that the current library is healthy.
- Supplement data: fill missing hashes, indexes, or metadata for older libraries.
- Duplicate scan: find duplicate, similar, and flipped images.
- Auto tag: generate tags in batch.
- Trash cleanup: permanently remove soft-deleted files after confirmation.

Recommended maintenance order:

1. Open the library.
2. Run the library check.
3. If missing data is reported, supplement it first.
4. Scan for duplicates.
5. Soft-delete confirmed unwanted duplicates.
6. Run automatic tagging.
7. Purge the trash only after confirming the result.

## 11. 数据结构 / Data Layout

### 中文

一个 `.library` 通常包含：

```text
YourProject.library/
  metadata.json
  tags.json
  catalog.db
  images/
    <UID>.info/
      metadata.json
      thumbnail.png
      <original file>
  index_semantic.faiss
  index_semantic.map
  index_sketch.faiss
  index_sketch.map
  index_sketch_fit.faiss
  index_sketch_fit.map
  index_sketch_crop.faiss
  index_sketch_crop.map
  .trash/
```

说明：

- `metadata.json`：图库基础信息。
- `tags.json`：标签数据。
- `catalog.db`：图片目录、哈希、路径、状态等结构化数据。
- `images/`：图库内部图片副本、缩略图和单图信息。每张图对应一个 `<UID>.info` 文件夹。
- `index_*.faiss`：搜索向量索引，例如语义索引和草图索引。
- `index_*.map`：索引行和图片 UID 的映射关系。
- `.trash/`：软删除回收区。

不建议手动编辑这些文件。需要迁移图库时，复制整个 `.library` 文件夹。

### English

A `.library` usually contains:

```text
YourProject.library/
  metadata.json
  tags.json
  catalog.db
  images/
    <UID>.info/
      metadata.json
      thumbnail.png
      <original file>
  index_semantic.faiss
  index_semantic.map
  index_sketch.faiss
  index_sketch.map
  index_sketch_fit.faiss
  index_sketch_fit.map
  index_sketch_crop.faiss
  index_sketch_crop.map
  .trash/
```

Description:

- `metadata.json`: basic library information.
- `tags.json`: tag data.
- `catalog.db`: structured image catalog, hashes, paths, status, and related data.
- `images/`: internal image copies, thumbnails, and per-image metadata. Each image has one `<UID>.info` folder.
- `index_*.faiss`: vector search indexes, such as semantic and sketch indexes.
- `index_*.map`: mappings between index rows and image UIDs.
- `.trash/`: soft-delete trash area.

Do not edit these files manually. To migrate a library, copy the entire `.library` folder.

## 12. 常见问题 / Troubleshooting

### 中文

**启动后浏览器没有打开**

手动访问：

```text
http://127.0.0.1:17788
```

**端口被占用**

换一个端口启动：

```bat
venv\Scripts\python.exe main.py --port 17888
```

**安装 CLIP 失败**

确认 Git 已安装，并重新运行：

```bat
setup.bat
```

**首次搜索或首次导入很慢**

这是正常现象。模型首次加载、索引首次建立都会更慢。

**语义搜索没有结果或结果不准**

尝试：

- 换成英文描述。
- 使用更具体的视觉词汇。
- 确认图片已完成导入和索引。
- 重新运行图库补全。

**重复扫描提示缺少数据**

先运行图库补全，再重新扫描重复图像。

**误删图片**

如果是在重复图像面板中删除，优先使用提示条中的 Undo，或从回收区恢复。清空回收区后无法通过 LocalMuse 恢复。

**程序异常退出**

查看：

```text
localmuse_crash.log
```

### English

**The browser does not open after launch**

Open manually:

```text
http://127.0.0.1:17788
```

**The port is already in use**

Start with another port:

```bat
venv\Scripts\python.exe main.py --port 17888
```

**CLIP installation failed**

Make sure Git is installed, then run:

```bat
setup.bat
```

**The first search or first import is slow**

This is normal. First model load and first index build take more time.

**Semantic search has no result or poor results**

Try:

- Use English descriptions.
- Use more specific visual terms.
- Make sure images have been fully imported and indexed.
- Run library supplement again.

**Duplicate scan reports missing data**

Run library supplement first, then scan again.

**Images were deleted by mistake**

If deletion happened in the duplicate panel, use the Undo toast first, or restore from the trash. After purging the trash, LocalMuse cannot restore those files.

**The app exits unexpectedly**

Check:

```text
localmuse_crash.log
```

## 13. 隐私说明 / Privacy

### 中文

LocalMuse 设计为本地图库工具：

- 图片导入、索引、搜索和标签数据默认在本机处理。
- 图库保存在用户选择的 `.library` 文件夹。
- 程序不会要求用户把图库上传到云端。
- 首次安装依赖或加载模型时，可能会访问外部依赖源或模型源。

如果项目图片涉及保密内容，请把 `.library` 文件夹放在受控磁盘位置，并避免把图库文件夹同步到公共云盘。

### English

LocalMuse is designed as a local library tool:

- Image import, indexing, search, and tag data are processed locally by default.
- Libraries are stored in user-selected `.library` folders.
- The app does not require uploading libraries to the cloud.
- During first dependency installation or first model loading, external package or model sources may be accessed.

For confidential project images, store the `.library` folder in a controlled disk location and avoid syncing it to public cloud drives.

## 开发者检查命令 / Developer Check Commands

### 中文

如需做基本代码健康检查，可以运行：

```bat
venv\Scripts\python.exe -m compileall -q main.py src
```

### English

For a basic code health check, run:

```bat
venv\Scripts\python.exe -m compileall -q main.py src
```
