import os

# 指定要遍歷的目錄
root_dir = "C:/nordlinglab-digitalupdrs-data/Cotracker_predicted_coords"

# 遍歷目錄
for subdir, dirs, files in os.walk(root_dir):
    print(len(subdir))
    for file in files:
        # 構建完整的檔案路徑
        file_path = os.path.join(subdir, file)
        
        # 檢查是否包含'HM'
        if 'HM' in file:
            # 刪除檔案
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

        elif 'RT' in file:
            new_file_name = file.replace('__RT', '_RT')
            new_file_path = os.path.join(subdir, new_file_name)
            os.rename(file_path, new_file_path)

        elif 'FT' in file:
            new_file_name = file.replace('__FT', '_FT')
            new_file_path = os.path.join(subdir, new_file_name)
            os.rename(file_path, new_file_path)

        elif 'TT' in file:
            new_file_name = file.replace('__TT', '_TT')
            new_file_path = os.path.join(subdir, new_file_name)
            os.rename(file_path, new_file_path)
        
        elif 'PT' in file:
            new_file_name = file.replace('__PT', '_PT')
            new_file_path = os.path.join(subdir, new_file_name)
            os.rename(file_path, new_file_path)

        if '.mp4' in file:
            # 移除檔案名中的'.mp4'
            new_file_name = file.replace('.mp4', '')
            new_file_path = os.path.join(subdir, new_file_name)
            # 重命名檔案
            os.rename(file_path, new_file_path)
            print(f"Renamed {file_path} to {new_file_path}")