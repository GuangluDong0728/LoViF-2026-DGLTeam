from PIL import Image
import os

# 设置输入输出文件夹
input_folder = "results/ALLINONE_TestLQ_FollowMedianTSE_1e_5_norotflip_p256_80000_meanWin16/visualization/Test_LQ"   # 输入文件夹路径
output_folder = "results/ALLINONE_TestLQ_FollowMedianTSE_1e_5_norotflip_p256_80000_meanWin16/visualization/Test_LQ_jpg" # 输出文件夹路径
# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 支持的图片格式
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

# 遍历文件夹
for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):
        img_path = os.path.join(input_folder, filename)
        # 输出文件名改为.jpg
        out_filename = os.path.splitext(filename)[0] + ".jpg"
        out_path = os.path.join(output_folder, out_filename)
        
        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                save_kwargs = dict(format="JPEG", quality=96, subsampling=0, optimize=True)
                img.save(out_path, **save_kwargs)
            print(f"✓ {filename}")
        except Exception as e:
            print(f"✗ {filename}: {e}")

