启动tmux
$ tmux 

退出
$ exit 或者 Ctrl + D 

启动命名tmux
$ tmux new -s <name>

分离会话
$ tmux detach

$ tmux ls可以查看当前所有的tmux为窗口

重接会话： 通过tmux detach关闭伪窗口后，希望再一次进入窗口
$ tmux attach -t 0  重新连接会话，使用伪窗口编号 
$ tmux attach -t xiaoqi 重新链接会话，使用伪窗口名称

杀死会话： 想要彻底杀死会话，不要再让它执行
$ tmux kill -session -t 0
$ tmux kill -session -t <name>

切换会话：
$ tmux switch -t 0
$ tmux switch -t <name>

重命名会话：
$ tmux rename-session -t 0 <new-name>

其他命令

$ tmux list-keys
$ tmux list-commonds
$ tmux info

