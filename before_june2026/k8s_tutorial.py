1. 创建任务:              kubectl create -f yaml_path -n xlab
2. 查看所有di job name:   kubectl get dijob -n xlab

3. 列出所有的pod name:    kubectl get pod -n xlab
4. 查看Pod的日志：         kubectl logs pod-name -n xlab

5. 删除任务：              kubectl delete dijob di-job-name -n xlab