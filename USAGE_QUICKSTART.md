# AI Devbox Quickstart

Active directory on the host:

```bash
/home/zhoutong/docker_share/ai-devbox
```

Shared workspace mounted into the container:

```bash
Host:      /home/zhoutong/docker_share
Container: /workspace
```

Preferred file exchange / project data directory:

```bash
Host:      /home/zhoutong/docker_share/docker_mapping
Container: /workspace/docker_mapping
```

## Daily Use

Start or enter the long-lived robotics container:

```bash
cd /home/zhoutong/docker_share/ai-devbox
./start-robotics.sh
```

Default long-lived container:

```bash
ai-robotics-devuser
```

Default robotics image:

```bash
local/ai-robotics:devuser-ros2-jazzy-mujoco-20260626
```

Default persistent home volume:

```bash
ai_robotics_devuser_home -> /home/dev
```

The container runs as the host UID/GID, so files created under bind-mounted folders should be editable from the host as `zhoutong`.

## Commands

### build-robotics.sh

Build the devuser robotics image from the existing stable robotics image. This does not overwrite the old root-based image tags.

```bash
cd /home/zhoutong/docker_share/ai-devbox
./build-robotics.sh
```

Use it when:

- first setting up this devuser image
- the devuser image was deleted
- Dockerfile devuser settings changed
- you intentionally want to refresh the devuser image

### start-robotics.sh

Preferred entry point for daily use.

```bash
cd /home/zhoutong/docker_share/ai-devbox
./start-robotics.sh
```

Behavior:

- if `ai-robotics-devuser` is running, enter it
- if it exists but is stopped, start and enter it
- if it does not exist, create and enter it

### exec-robotics.sh

Enter the existing long-lived robotics container.

```bash
cd /home/zhoutong/docker_share/ai-devbox
./exec-robotics.sh
```

Use this for extra terminal windows after the container already exists. If the container does not exist, run `./start-robotics.sh` first.

### shell-robotics.sh

Start a temporary one-shot robotics container.

```bash
cd /home/zhoutong/docker_share/ai-devbox
./shell-robotics.sh
```

The container is removed when you exit. Prefer `./start-robotics.sh` for normal work.

### install-ai-tools.sh

Run inside the container to install Codex and Claude Code into the persistent `dev` home.

```bash
cd /workspace/ai-devbox
./install-ai-tools.sh
source ~/.bashrc
```

Installed location:

```bash
/home/dev/.npm-global
```

After installation:

```bash
codex --version
claude --version
claude --dangerously-skip-permissions


codex --sandbox danger-full-access
```

### fix-permissions.sh

Run on the host to repair ownership of shared folders if files become hard to edit outside Docker.

```bash
cd /home/zhoutong/docker_share/ai-devbox
./fix-permissions.sh
```

This fixes ownership under:

```bash
/home/zhoutong/docker_share
/home/zhoutong/docker_share/docker_mapping
/home/zhoutong/docker_share/ai-devbox
```

Avoid using `sudo` inside shared project folders unless necessary. Files created with `sudo` may become root-owned again.

## Multiple Windows

Yes, multiple terminal windows are supported.

Option 1: run the same command in each terminal:

```bash
cd /home/zhoutong/docker_share/ai-devbox
./start-robotics.sh
```

Option 2: start once, then exec from other terminals:

```bash
cd /home/zhoutong/docker_share/ai-devbox
./start-robotics.sh
```

Other terminals:

```bash
cd /home/zhoutong/docker_share/ai-devbox
./exec-robotics.sh
```

Both enter the same `ai-robotics-devuser` container.

## Useful Checks

Inside the container:

```bash
whoami
echo "$HOME"
id
```

Expected:

```text
dev
/home/dev
uid=1000 gid=1000
```

Check Docker-created file ownership from the host:

```bash
stat -c '%U:%G %a %n' /home/zhoutong/docker_share/docker_mapping
```

Expected owner:

```text
zhoutong:zhoutong
```
