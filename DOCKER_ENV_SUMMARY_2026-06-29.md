# Docker Environment Summary - 2026-06-29

This file summarizes the current AI/robotics Docker environment in
`/home/zhoutong/docker_share/ai-devbox`.

## Current Status

The Docker setup is now considered stable enough for daily use.

Current active container:

```bash
ai-robotics-devuser
```

Current active image:

```bash
local/ai-robotics:devuser-ros2-jazzy-mujoco-20260626
```

Current persistent home volume:

```bash
ai_robotics_devuser_home -> /home/dev
```

Daily entry point:

```bash
cd /home/zhoutong/docker_share/ai-devbox
./scripts/start-robotics.sh
```

This command will:

- enter `ai-robotics-devuser` if it is already running
- start and enter it if it exists but is stopped
- create and enter it if it does not exist

## Project Layout

The project root is organized by file type:

```text
/home/zhoutong/docker_share/ai-devbox/
  docs/      Markdown documentation and maintenance notes.
  docker/    Dockerfiles and docker-compose.yml.
  scripts/   Host/container operation scripts.
```

Recommended first file for future agents:

```bash
/home/zhoutong/docker_share/ai-devbox/docs/DOCKER_ENV_SUMMARY_2026-06-29.md
```

Then read these if more detail is needed:

```bash
/home/zhoutong/docker_share/ai-devbox/docs/USAGE_QUICKSTART.md
/home/zhoutong/docker_share/ai-devbox/docs/AI_ROBOTICS_DOCKER_GUIDE.md
```

Current scripts should be run from the project root with the `scripts/` prefix.
They compute project paths from their own location, so they are not sensitive to
the caller's current working directory.

## Images

Main images currently relevant to this environment:

```text
local/ai-robotics:devuser-ros2-jazzy-mujoco-20260626
local/ai-robotics:devuser-ros2-jazzy-mujoco
local/ai-robotics:stable-ros2-jazzy-mujoco-20260626
local/ai-robotics:ros2-jazzy-mujoco
local/ai-devbox:ubuntu24
ubuntu:24.04
portainer/portainer-ce:lts
```

The image used for daily work is:

```bash
local/ai-robotics:devuser-ros2-jazzy-mujoco-20260626
```

It contains:

- Ubuntu 24.04
- ROS2 Jazzy desktop
- MuJoCo Python package 3.9.0
- Node.js 22
- Python 3.12
- common development tools such as git, curl, vim, tmux, jq, and ripgrep

The older `stable-ros2-jazzy-mujoco-20260626` image is the original robotics base.
The newer `devuser-ros2-jazzy-mujoco-20260626` image adds host UID/GID alignment
for the `dev` user, which makes shared-file permissions easier to manage.

## Shell Scripts

Daily-use scripts:

```text
scripts/start-robotics.sh        Preferred entry point for daily use.
scripts/exec-robotics.sh         Enter an existing long-lived robotics container.
scripts/shell-robotics.sh        Start a temporary one-shot robotics container.
scripts/install-ai-tools.sh      Install Codex and Claude CLI inside /home/dev.
scripts/test-robotics.sh         Verify ROS2 and MuJoCo.
scripts/fix-permissions.sh       Repair ownership of shared folders from the host.
```

Build scripts:

```text
scripts/build-robotics.sh        Build the current devuser robotics image.
scripts/build-robotics-clean.sh  Clean builder state, then rebuild devbox and robotics images.
scripts/build.sh                 Build the regular ai-devbox devuser image.
scripts/build-clean.sh           Clean builder state, pull ubuntu:24.04, then run build.sh.
```

Regular devbox and maintenance scripts:

```text
scripts/start.sh                 Start the regular ai-devbox service with docker compose.
scripts/exec.sh                  Enter the regular ai-devbox service.
scripts/shell.sh                 Start a one-shot regular ai-devbox container.
scripts/snapshot.sh              Commit the running ai-devbox container to a snapshot image.
scripts/test-network.sh          Test Docker registry, OpenAI, and Anthropic connectivity.
scripts/entrypoint.sh            Container entrypoint for home setup and Docker socket group handling.
```

For normal robotics, ROS2, MuJoCo, Codex, and Claude work, prefer:

```bash
./scripts/start-robotics.sh
```

## Shared Folders And Permissions

The shared folder behavior is implemented with Docker bind mounts.

Main workspace mapping:

```text
Host:      /home/zhoutong/docker_share
Container: /workspace
```

Preferred file exchange and project data mapping:

```text
Host:      /home/zhoutong/docker_share/docker_mapping
Container: /workspace/docker_mapping
```

The robotics scripts also mount the persistent Docker volume:

```text
Volume:    ai_robotics_devuser_home
Container: /home/dev
```

`/home/dev` stores user state inside Docker, including:

- Codex login files under `/home/dev/.codex`
- Claude login/config files under `/home/dev/.claude`
- npm global packages under `/home/dev/.npm-global`
- shell config files and other user config

The container is started with the host UID/GID:

```bash
--user "${HOST_UID}:${HOST_GID}"
```

This is why files created under `/workspace` or `/workspace/docker_mapping`
should normally be editable from both the host and the container.

If shared files become owned by root or hard to edit from the host, run:

```bash
cd /home/zhoutong/docker_share/ai-devbox
./scripts/fix-permissions.sh
```

Avoid using `sudo` inside shared project folders unless necessary, because it can
create root-owned files on the host-mounted directory.

## Codex And Claude Installation

Codex and Claude CLI are intentionally not baked into the image. They are installed
inside the persistent `/home/dev` volume so they can be updated independently from
the Docker image.

Enter the container:

```bash
cd /home/zhoutong/docker_share/ai-devbox
./scripts/start-robotics.sh
```

Install both tools:

```bash
cd /workspace/ai-devbox
./scripts/install-ai-tools.sh
source ~/.bashrc
```

Check installation:

```bash
codex --version
claude --version
```

Claude normal use:

```bash
claude
```

Claude bypass-permissions mode:

```bash
claude --dangerously-skip-permissions
```

This works because the container enters as the non-root `dev` user.

## Codex Login

Recommended login method inside the container:

```bash
codex login --device-auth
```

Then check status:

```bash
codex login status
```

Alternative API-key login:

```bash
printenv OPENAI_API_KEY | codex login --with-api-key
codex login status
```

Codex login state is stored under:

```bash
/home/dev/.codex
```

Because `/home/dev` is backed by the persistent Docker volume
`ai_robotics_devuser_home`, the login state should survive container stop/start
and container recreation.

Codex may require login again if:

- `ai_robotics_devuser_home` is deleted
- a different home volume is used
- `/home/dev/.codex` is manually removed
- the token expires or the account requires reauthentication

Do not put Codex tokens, OpenAI API keys, or Claude credentials into Dockerfiles
or committed snapshot images.

## Proxy Notes

The scripts pass common proxy variables into the container:

```text
http_proxy
https_proxy
HTTP_PROXY
HTTPS_PROXY
no_proxy
NO_PROXY
```

If Codex login fails during token exchange, check proxy variables inside the
container:

```bash
env | grep -i proxy
```

If needed, temporarily set them:

```bash
export http_proxy="${http_proxy:-http://127.0.0.1:6789/}"
export https_proxy="${https_proxy:-http://127.0.0.1:6789/}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
export NO_PROXY="$no_proxy"
```

Test OpenAI auth endpoint reachability:

```bash
curl -sS -o /dev/null -w "auth=%{http_code}\n" https://auth.openai.com/oauth/token
```

`auth=405` is acceptable. It means the endpoint is reachable, but the request
method is not accepted for that URL.

## Documentation Note

Some older documents still mention old names such as:

```text
ai-robotics
ai_robotics_home
local/ai-robotics:stable-ros2-jazzy-mujoco-20260626
```

The current daily-use names are:

```text
Container: ai-robotics-devuser
Volume:    ai_robotics_devuser_home
Image:     local/ai-robotics:devuser-ros2-jazzy-mujoco-20260626
```

Use the current scripts as the source of truth when there is a mismatch.
