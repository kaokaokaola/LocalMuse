# 新建 GitHub 项目并上传 LocalMuse V2 — 操作指南

本指南配合本目录下新增的 `.gitignore`、`LICENSE`、`README.md` 使用。

## 0. 清理一个小问题

我在沙箱里测试 git 时,意外在本目录下生成了两个空文件夹:`.git/` 和
`testdir123/`。沙箱权限无法从这边删除它们,**请在 Windows 资源管理器里手动删掉这两个文件夹**(都是空的/无用的,删了不影响任何东西)。

## 1. 上传清单 — 哪些会被包含,哪些会被排除

`.gitignore` 已经配置好,执行 `git add .` 时会自动按下表处理,你不需要手动逐个挑选:

| 内容 | 是否上传 | 说明 |
|---|---|---|
| `src/`、`main.py`、`LocalMuse.bat` | ✅ 上传 | 核心代码 |
| `setup.bat`、`setup_annotation.bat`、`install_flash_attn.bat`、`build_portable_python.bat`、`cleanup_venv.bat` | ✅ 上传 | 安装/打包脚本 |
| `requirements.txt`、`requirements-optional.txt` | ✅ 上传 | 依赖清单 |
| `README.md`、`LICENSE`、`.gitignore`、`LOCALMUSE_MODIFICATION_GUIDE.md`、`PORTABLE_DEPLOYMENT.md` | ✅ 上传 | 文档 |
| `address.txt` | ⚠️ 上传前检查一下 | 看看里面是否有本机绝对路径或其它你不想公开的内容,没问题就保留 |
| `test_search.py` | ✅ 上传 | 测试脚本 |
| `venv/` (~6GB) | ❌ 自动排除 | 不可移植,用户跑 `setup.bat` 自动生成 |
| `python/` (~6GB) | ❌ 自动排除 | 便携 Python,体积太大,走 Release 单独分发 |
| `__pycache__/`、`*.log` | ❌ 自动排除 | 运行时产物 |
| `landing/`、`landing.7z` | 视情况 | 这是 landing page,可以放进这个仓库的 `docs/` 目录用 GitHub Pages 发布,或放到独立仓库——见下方"可选"部分 |

## 2. 在 GitHub 网站上新建仓库

1. 登录 GitHub → 右上角 `+` → **New repository**。
2. Repository name:例如 `localmuse`。
3. Description(可选):`Local multimodal image search for designers — semantic, sketch, depth & pose, 100% offline.`
4. Public(开源推荐选 Public,这样才能被引用/被搜到)。
5. **不要**勾选 "Add a README file" / "Add .gitignore" / "Choose a license"——本地已经准备好了,勾选会导致后面 push 冲突。
6. 点击 **Create repository**。创建后页面会显示一个仓库地址,形如:
   ```
   https://github.com/<你的用户名>/localmuse.git
   ```

## 3. 在本机(Windows)执行 git 命令

在项目根目录(`2026.05.24.LocalMuse.V2`)打开终端(PowerShell 或 cmd):

```bat
cd C:\Users\Administrator\2026.05.24.LocalMuse.V2

:: 如果之前没初始化过 git
git init
git branch -M main

:: 配置你的身份(只需一次,全局生效)
git config --global user.name "你的名字"
git config --global user.email "S4208097@student.rmit.edu.au"

:: 添加远程仓库地址(替换成你自己的)
git remote add origin https://github.com/<你的用户名>/localmuse.git

:: 检查将要提交的文件 —— 重点确认没有 venv/、python/
git add .
git status

:: 提交并推送
git commit -m "Initial commit: LocalMuse V2"
git push -u origin main
```

`git status` 这一步很重要:如果看到 `venv/` 或 `python/` 出现在待提交列表里,
说明 `.gitignore` 没生效(通常是因为这两个文件夹在 `.gitignore` 写好之前已经被
`git add` 过)。遇到这种情况执行:

```bat
git rm -r --cached venv python
```

再重新 `git add .` 一次即可。

## 4. 首次推送如果要求登录

GitHub 已不支持密码登录 push,需要用 **Personal Access Token (PAT)** 代替密码:

1. GitHub 右上角头像 → Settings → Developer settings → Personal access tokens
   → Tokens (classic) → Generate new token。
2. 勾选 `repo` 权限,生成后**复制保存**(只显示一次)。
3. `git push` 弹出登录框时,用户名填你的 GitHub 用户名,密码填这个 token。

(也可以改用 GitHub Desktop 客户端,图形界面操作,避免命令行。)

## 5. 可选:发布 landing page (GitHub Pages)

如果想把之前生成的 `landing/index.html` 也放进这个仓库并通过
`https://<用户名>.github.io/localmuse/` 访问:

```bat
mkdir docs
move landing\index.html docs\index.html
git add docs
git commit -m "Add project landing page"
git push
```

然后在 GitHub 仓库页面:Settings → Pages → Source 选择 `main` 分支 /
`/docs` 目录 → Save。几分钟后即可访问。

## 6. 上传后检查

- 打开仓库页面,确认看不到 `venv/`、`python/`、`__pycache__/`。
- 仓库总大小应该在几 MB 级别(代码 + 文档),不应该是几 GB。
- 点开 `README.md` 确认渲染正常,Quick start 命令准确。
- 给仓库加 Topics(标签):`image-search`、`clip`、`vector-search`、
  `multimodal`、`design-tools` 等,方便被搜索到。
- 如果要单独分发便携版 `python\`(打包后),在仓库页面右侧 **Releases** →
  Draft a new release → 上传压缩包(注意单文件 2GB 限制,超过需分卷或放外部链接)。
