import os
import re
import shutil

# 判断是否是录音文件
def is_record(name):
    if re.match("^[0-9]{11}[-_][0-9]{2}[-_][0-9]{2}\.dat$", name):
        return True
    return False

# 递归获得path中所有录音文件
def dfs_dir(path):
    curr = os.listdir(path)
    files = []
    for item in curr:
        npath = os.path.join(path, item)
        if os.path.isdir(npath):
            files += dfs_dir(npath)
        elif os.path.isfile(npath):
            if is_record(item):
                files.append(npath)
    return files

# 整理数据文件为规范格式
def collate_data():
    root_path = "data_after512"
    new_root_path = "dataset"
    if not os.path.exists(new_root_path):
        os.makedirs(new_root_path)
    dirs = os.listdir(root_path)
    # 遍历源文件夹，依次处理
    for dir in dirs:
        student_id = dir[:11]
        print("正在处理：", student_id)
        new_path = os.path.join(new_root_path, student_id)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        
        # 列出所有文件
        path = os.path.join(root_path, dir)
        voices = dfs_dir(path)

        # 检查文件个数为400
        file_num = len(voices)
        if file_num != 400:
            print(f"{path} has {file_num} files")

        # print(os.path.basename(voices[0]))
        id, word, cnt = re.split(r'[-_]', os.path.basename(voices[0]))
        # print(id, word, cnt)
        add = 0
        if word == "00":
            add = 1

        # 拷贝到新文件夹
        for src in voices:
            id, word, cnt = re.split(r'[-_]', os.path.basename(src))
            new_name = f"{id}_{int(word) + add:02d}_{cnt}"
            dst = os.path.join(new_path, new_name)
            shutil.copy(src, dst)

if __name__ == "__main__":
    collate_data()
