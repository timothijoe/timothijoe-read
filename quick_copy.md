# Quick Copy

集中存放日常使用、方便直接复制的命令。

## Enter the container

```bash
cd /home/zhoutong/docker_share/ai-devbox
./scripts/start-robotics.sh
```

## After installation

```bash
codex --version
claude --version
claude --dangerously-skip-permissions


codex --sandbox danger-full-access
```


## 2026-07-19-185542-NoMachine远程连接与SSH隧道排查-summary.md
## 远端连接方案

远端主机通过 SSH 地址 `175.155.64.171:24106`、用户 `linux` 访问。为了避免把 NoMachine 的 `4000` 端口直接暴露到公网，采用本机端口 `14000` 到远端回环地址 `127.0.0.1:4000` 的 SSH 转发：

```bash
ssh -N \
  -L 127.0.0.1:14000:127.0.0.1:4000 \
  -p 24106 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  linux@175.155.64.171
```

该命令必须在本机 `zhoutong-System-Product-Name` 上运行，而不是在远端 `ubuntu-9766` 中运行。隧道建立后，NoMachine 应使用 NX 协议连接 `127.0.0.1:14000`。

## NoMachine 客户端配置

SSH 隧道成功建立后，保持运行隧道的本机终端开启。打开本机 NoMachine，点击左上角 **Add**，再选择 **Add connection**，按以下参数创建连接：

```text
Name: 远端 Ubuntu
Host: 127.0.0.1
Port: 14000
Protocol: NX
```

## Git

```bash
git status
```

```bash
git add .
git commit -m "update"
git push
```

## 临时指令

<!-- 把需要快速复制的新指令添加到这里。 -->
