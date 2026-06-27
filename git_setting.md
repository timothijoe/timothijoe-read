# timothijoe-read

各种教程、笔记与代码片段的集合仓库。

---

## 🔐 GitHub SSH 配置指南（免 Token 推送）

> 推荐使用 SSH 方式连接 GitHub，配置一次，以后 `git push` 再也不用输 Token。

### 方法一：SSH（推荐 ✅）

#### 第 1 步：检查是否已有 SSH Key

```bash
ls ~/.ssh
```

如果看到 `id_ed25519` + `id_ed25519.pub`，说明已经有了，跳到第 5 步。没有就继续往下走。

#### 第 2 步：生成 SSH Key

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

例如：

```bash
ssh-keygen -t ed25519 -C "abc@gmail.com"
```

- 提示 `Enter file in which to save the key` → 直接回车
- 提示 `Enter passphrase` → 直接回车（一路回车即可）

生成后会在 `~/.ssh/` 下得到两个文件：

```
~/.ssh/id_ed25519       # 私钥（自己留着，不要给任何人）
~/.ssh/id_ed25519.pub   # 公钥（给 GitHub 的）
```

#### 第 3 步：启动 ssh-agent 并添加密钥

```bash
# 启动 ssh-agent（Ubuntu）
eval "$(ssh-agent -s)"

# 添加私钥
ssh-add ~/.ssh/id_ed25519
```

看到 `Identity added` 就说明成功了。

#### 第 4 步（可选）：配置 SSH 走 443 端口

有些网络环境（公司防火墙、学校网络）会封 22 端口，导致 `ssh -T git@github.com` 连不上。配置 SSH 走 443 端口即可绕过：

```bash
nano ~/.ssh/config
```

加入以下内容：

```
Host github.com
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/id_ed25519
  ProxyCommand nc -X connect -x 127.0.0.1:6789 %h %p
```

> 💡 **说明**：`ssh.github.com` 是 GitHub 专门为 443 端口提供的 SSH 端点。配置后所有 `git@github.com` 的连接会自动走 443 端口，无需改仓库地址。

保存后测试：

```bash
ssh -T git@github.com
```

#### 第 5 步：复制公钥 🔑

```bash
cat ~/.ssh/id_ed25519.pub
```

会输出类似这样的内容：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI...... 你的邮箱
```

**把这一整串全部复制下来。**

#### 第 6 步：在 GitHub 上添加 SSH Key

1. 登录 [GitHub](https://github.com)
2. 右上角头像 → **Settings**
3. 左侧菜单 → **SSH and GPG keys**
4. 点击 **New SSH key**
5. **Title**：随便填，比如 `Ubuntu24`
6. **Key**：粘贴刚才复制的那串公钥
7. 点击 **Add SSH key** 保存

#### 第 7 步：测试连接

```bash
ssh -T git@github.com
```

第一次会提示：

```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入 `yes` 回车。如果看到：

```
Hi xxx! You've successfully authenticated...
```

说明 SSH 已配置成功 🎉

#### 第 8 步：把仓库地址从 HTTPS 改成 SSH

很多人的仓库还是 HTTPS 地址，需要改成 SSH。

先看看当前用的什么：

```bash
git remote -v
```

如果看到：

```
origin  https://github.com/xxx/xxx.git (fetch)
origin  https://github.com/xxx/xxx.git (push)
```

改成 SSH：

```bash
git remote set-url origin git@github.com:用户名/仓库.git
```

例如：

```bash
git remote set-url origin git@github.com:sun/myproject.git
```

再确认一下：

```bash
git remote -v
```

现在应该变成：

```
origin  git@github.com:xxx/xxx.git (fetch)
origin  git@github.com:xxx/xxx.git (push)
```

#### 第 9 步：推送

以后只需：

```bash
git push
```

**完全不需要 Token。永远不需要。**

---

> 💡 **为什么用 ed25519？** 比 RSA 更安全、更快、密钥更短，是现代默认推荐。

---

## 📁 仓库内容

本仓库包含各种学习笔记、教程脚本和小工具：

### 深度学习 / PyTorch

| 文件 | 说明 |
|------|------|
| `load_pytorch.py` | PyTorch 模型加载基础 |
| `load_pytorch_tutorial.py` | PyTorch 模型加载详细教程 |

### 树莓派（Raspberry Pi）

| 文件 | 说明 |
|------|------|
| `rpi.py` | 树莓派相关脚本 |
| `rpi_data_preprocess_pipeline.py` | 数据预处理流水线 |

### Linux / 系统工具

| 文件 | 说明 |
|------|------|
| `system.py` | 系统操作脚本 |
| `system_conmmand.py` | 系统命令记录 |
| `tmux_tutorial.py` | Tmux 教程 |
| `k8s_tutorial.py` | Kubernetes 教程笔记 |
| `make_path_tutorial.py` | Makefile/Path 教程 |

### 其他

| 文件 | 说明 |
|------|------|
| `hunter.py` / `hunter_project.py` | Hunter 项目脚本 |
| `march2023.py` | 2023 年 3 月笔记 |
| `april.py` | 四月笔记 |
| `october.py` | 十月笔记 |
| `august_update.md` | 八月更新记录 |
| `conv_tutorial.py` | 卷积教程 |

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone git@github.com:timothijoe/timothijoe-read.git

# 进入目录
cd timothijoe-read
```

---

*最后更新：2026 年 6 月*
