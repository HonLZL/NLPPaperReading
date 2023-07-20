## 1 登录并查看服务器情况

**登录** 

ssh AiUser@192.168.0.100

密码：admin



**管理员权限**

sudo su

密码：admin



**查看服务器系统与配置** 

cat /proc/version  查看系统版本

lscpu  查看cpu信息

Model name: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz

lsblk  查看硬盘信息

free -h 显示内存单位，free -m 显示内存使用情况，cat /proc/meminfo 查看内存详细信息

lspci | grep VGA  查看显卡信息

nvidia-smi 查看显卡使用情况



