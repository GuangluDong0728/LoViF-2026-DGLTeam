import os

def simple_rename_x4_suffix(folder_path):
    """
    简单版本：直接去掉文件名中的x4后缀
    
    Args:
        folder_path (str): LR图片文件夹路径
    """
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return
    
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    renamed_count = 0
    
    print(f"开始处理文件夹：{folder_path}")
    print("-" * 50)
    
    for filename in files:
        # 检查文件是否包含x4
        if '_ALLINONE_TestLQ_FollowMedianTSE_1e_5_norotflip_p256_80000_meanWin16' in filename:
            # 构建完整的文件路径
            old_path = os.path.join(folder_path, filename)
            
            # 生成新文件名（去掉x4）
            new_filename = filename.replace('_ALLINONE_TestLQ_FollowMedianTSE_1e_5_norotflip_p256_80000_meanWin16', '')
            new_path = os.path.join(folder_path, new_filename)
            
            # 检查新文件名是否已存在
            if os.path.exists(new_path):
                print(f"跳过 {filename} -> 目标文件已存在")
                continue
            
            try:
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"✓ {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"✗ 重命名失败 {filename}: {e}")
    
    print("-" * 50)
    print(f"完成！成功重命名 {renamed_count} 个文件")

# 使用方法：修改下面的路径为你的实际路径
if __name__ == "__main__":
    # 方法1：直接指定路径
    lr_folder = "results/ALLINONE_TestLQ_FollowMedianTSE_1e_5_norotflip_p256_80000_meanWin16/visualization/Test_LQ"  # 修改为你的实际路径
    simple_rename_x4_suffix(lr_folder)
    
    # 方法2：交互式输入路径
    # folder_path = input("请输入LR图片文件夹路径: ").strip()
    # simple_rename_x4_suffix(folder_path)

# import os
# import re
# import glob
# from pathlib import Path

# def remove_prefix_from_filenames(directory, prefix_pattern=None, dry_run=True):
#     """
#     批量重命名文件，去掉指定前缀
    
#     参数:
#     directory: 文件所在目录
#     prefix_pattern: 前缀模式，如果为None则自动检测
#     dry_run: 是否为试运行（True=只显示预览，False=实际重命名）
#     """
    
#     # 转换为Path对象
#     directory = Path(directory)
    
#     if not directory.exists():
#         print(f"目录不存在: {directory}")
#         return
    
#     # 查找所有PNG文件
#     png_files = list(directory.glob("*.png"))
    
#     if not png_files:
#         print(f"在 {directory} 中未找到PNG文件")
#         return
    
#     print(f"找到 {len(png_files)} 个PNG文件")
    
#     # 如果没有指定模式，尝试自动检测
#     if prefix_pattern is None:
#         # 分析文件名模式，提取通用前缀
#         prefix_pattern = detect_prefix_pattern(png_files)
#         if prefix_pattern:
#             print(f"自动检测到前缀模式: {prefix_pattern}")
#         else:
#             print("无法自动检测前缀模式，请手动指定")
#             return
    
#     # 使用正则表达式匹配和重命名
#     pattern = re.compile(prefix_pattern)
#     renamed_count = 0
    
#     print(f"\n{'模式' if dry_run else '执行'}重命名:")
#     print("-" * 50)
    
#     for file_path in sorted(png_files):
#         filename = file_path.name
#         match = pattern.match(filename)
        
#         if match:
#             # 提取数字部分
#             number_part = match.group(1)
#             new_filename = f"{number_part}.png"
#             new_path = file_path.parent / new_filename
            
#             # 检查新文件名是否已存在
#             if new_path.exists() and new_path != file_path:
#                 print(f"⚠️  跳过 {filename} -> {new_filename} (目标文件已存在)")
#                 continue
            
#             if dry_run:
#                 print(f"📝 {filename} -> {new_filename}")
#             else:
#                 try:
#                     file_path.rename(new_path)
#                     print(f"✅ {filename} -> {new_filename}")
#                     renamed_count += 1
#                 except Exception as e:
#                     print(f"❌ 重命名失败 {filename}: {e}")
#         else:
#             print(f"⚠️  跳过 {filename} (不匹配模式)")
    
#     if dry_run:
#         print(f"\n试运行完成！预计重命名 {len([f for f in png_files if pattern.match(f.name)])} 个文件")
#         print("使用 dry_run=False 执行实际重命名")
#     else:
#         print(f"\n重命名完成！成功重命名 {renamed_count} 个文件")

# def detect_prefix_pattern(file_list):
#     """
#     自动检测文件名前缀模式
#     """
#     # 获取所有文件名
#     filenames = [f.name for f in file_list]
    
#     # 查找常见的数字模式
#     patterns = [
#         r'.*?(\d{4})\.png$',  # 4位数字
#         r'.*?(\d{3})\.png$',   # 3位数字
#         r'.*?(\d{5})\.png$',   # 5位数字
#         r'.*?(\d+)\.png$',     # 任意位数字
#     ]
    
#     for pattern in patterns:
#         regex = re.compile(pattern)
#         matches = [regex.match(filename) for filename in filenames]
        
#         # 如果大部分文件都匹配这个模式
#         if sum(1 for m in matches if m) > len(filenames) * 0.8:
#             return pattern
    
#     return None

# def rename_with_custom_pattern(directory, old_pattern, new_pattern, dry_run=True):
#     """
#     使用自定义模式重命名文件
    
#     参数:
#     directory: 文件目录
#     old_pattern: 旧文件名正则模式 (使用捕获组)
#     new_pattern: 新文件名模式 (可以使用 \\1, \\2 等引用捕获组)
#     dry_run: 是否试运行
    
#     示例:
#     old_pattern = r'poled_test_gt_(\d{4})\.png'
#     new_pattern = r'\\1.png'
#     """
    
#     directory = Path(directory)
#     png_files = list(directory.glob("*.png"))
    
#     pattern = re.compile(old_pattern)
#     renamed_count = 0
    
#     print(f"使用自定义模式重命名:")
#     print(f"匹配模式: {old_pattern}")
#     print(f"新名称模式: {new_pattern}")
#     print("-" * 50)
    
#     for file_path in sorted(png_files):
#         filename = file_path.name
#         match = pattern.match(filename)
        
#         if match:
#             # 使用正则替换生成新文件名
#             new_filename = pattern.sub(new_pattern, filename)
#             new_path = file_path.parent / new_filename
            
#             if new_path.exists() and new_path != file_path:
#                 print(f"⚠️  跳过 {filename} -> {new_filename} (目标文件已存在)")
#                 continue
            
#             if dry_run:
#                 print(f"📝 {filename} -> {new_filename}")
#             else:
#                 try:
#                     file_path.rename(new_path)
#                     print(f"✅ {filename} -> {new_filename}")
#                     renamed_count += 1
#                 except Exception as e:
#                     print(f"❌ 重命名失败 {filename}: {e}")
#         else:
#             print(f"⚠️  跳过 {filename} (不匹配模式)")
    
#     if not dry_run:
#         print(f"\n重命名完成！成功重命名 {renamed_count} 个文件")

# def preview_files(directory, pattern=None):
#     """预览目录中的文件名"""
#     directory = Path(directory)
#     png_files = sorted(directory.glob("*.png"))
    
#     print(f"目录: {directory}")
#     print(f"PNG文件数量: {len(png_files)}")
#     print("-" * 50)
    
#     # 显示前10个和后5个文件名
#     preview_files = png_files[:10] + (['...'] if len(png_files) > 15 else []) + png_files[-5:]
    
#     for i, file_path in enumerate(preview_files):
#         if file_path == '...':
#             print("...")
#             continue
#         filename = file_path.name if hasattr(file_path, 'name') else str(file_path)
        
#         if pattern:
#             match = re.match(pattern, filename)
#             if match:
#                 print(f"{filename} -> {match.group(1)}.png")
#             else:
#                 print(f"{filename} (不匹配)")
#         else:
#             print(filename)

# # 使用示例
# if __name__ == "__main__":
#     # 示例1: 自动检测并重命名 (推荐)
#     remove_prefix_from_filenames('/mnt/dgl/Flare7Kapp/test_data/synthetic/input', dry_run=False)
    
#     # 示例2: 指定具体模式
#     # remove_prefix_from_filenames('/path/to/your/images', 
#     #                             prefix_pattern=r'poled_test_gt_(\d{4})\.png$', 
#     #                             dry_run=True)
    
#     # 示例3: 使用自定义模式 (最灵活)
#     # rename_with_custom_pattern('/path/to/your/images',
#     #                           old_pattern=r'poled_test_gt_(\d{4})\.png',
#     #                           new_pattern=r'\1.png',
#     #                           dry_run=True)
    
#     # 示例4: 预览文件
#     # preview_files('/path/to/your/images', r'poled_test_gt_(\d{4})\.png$')
    
#     # print("批量重命名工具已准备就绪！")
#     # print("\n快速使用:")
#     # print("1. 自动重命名: remove_prefix_from_filenames('目录路径')")
#     # print("2. 自定义模式: rename_with_custom_pattern('目录', r'old_pattern', r'new_pattern')")
#     # print("3. 预览文件: preview_files('目录路径')")
#     # print("\n⚠️  首次使用建议设置 dry_run=True 先预览效果！")