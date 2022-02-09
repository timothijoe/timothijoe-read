import os


# 得到当前文件夹下所有的子文件夹名称
def scandir(path):
    filelist = os.listdir(path)
    dir_list = []
    for filename in filelist:
        # 获得子文件夹的绝对路径
        filepath = os.path.join(path,filename)
        if os.path.isdir(filepath):
            dir_list.append(filepath)
    return dir_list

# 得到当前文件夹 (d_name)下的所有路径
def get_file_name(d_name):
    file_list = []
    for cur_file in os.listdir(d_name):
        cur_bag_dir = os.path.join(d_name, cur_file)
        file_list.append(cur_bag_dir)
    return file_list

# 如果new_folder_path不存在，则创建
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# 获得当前文件夹内的文件夹数量，用于命名方便
pwd = os.getcwd()
# 当前文件夹的上一级目录
father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")


target_folder_dir = pwd
folder_list = scanfile(target_folder_dir)
folder_num = len(folder_list)
new_folder_name = "folder%02i_data" % folder_num
new_folder_path = os.path.join(target_folder_dir, new_folder_name)
print(new_folder_path)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)



def mk_logdir(params):
    path1 = 'result'
    path2 = 'result/{}/ckpt'.format(params.exp_name)
    path3 = 'result/{}/log'.format(params.exp_name)
    path4 = 'result/{}/images'.format(params.exp_name)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3):
        os.makedirs(path3)
    if not os.path.exists(path4):
        os.makedirs(path4)

关于读取一个文件，后把这个文件的名字作为另一个文件名字或者名字的一部分

os.path.split('PATH') ： 
（1） PATH是指一个文件的全路径作为参数
（2） 如果给出的是一个目录和文件名，则输出路径和文件名
（3） 如果给出的是一个目录名，则输出路径和为空的文件名
实际上，该函数的分割并不只能，仅仅以PATH中最后一个'/'作为分隔负号，分隔后，将索引为0的视为目录（路径——，将索引1视作文件名

使用basename()函数：
import ps.path
filePATH = '/home/hello.txt'
x = os.path.basename(filePATH)

' x = hello.txt'
去后缀：
os.path.splittext(x)[0]
 则分离文件名字和扩展名；返回默认(fname, fextension)元组，若[1]则返回后缀
