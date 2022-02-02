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
target_folder_dir = pwd
folder_list = scanfile(target_folder_dir)
folder_num = len(folder_list)
new_folder_name = "folder%02i_data" % folder_num
new_folder_path = os.path.join(target_folder_dir, new_folder_name)
print(new_folder_path)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)