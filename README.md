# [LoViF2026] - [The First Challenge on Real-World All-in-One Image Restoration] - [DGLTeam]

# Environment Prepare
You can refer to the environment preparation process of [BasicSR](https://github.com/XPixelGroup/BasicSR), which mainly includes the following two steps:

1. 

    ```bash
    pip install -r requirements.txt
    ```

2. 

    ```bash
    python setup.py develop
    ```

# Downloading Our Weights

1. **Download Pretrained Weights:**
   - Navigate to [this link](https://drive.google.com/file/d/1wz3YGhhOd2dV6C6_1XBKmsEwxsowgomV/view?usp=drive_link) to download our weights. 

2. **Save to `experiments` Directory:**
   - Once downloaded, place the weights into the `experiments` directory.
     
# Testing

To test our model, please open the `options/NTIRE2026/test_se.yml` file and update the paths, and just run the command:

```bash
python basicsr/test.py -opt options/NTIRE2026/test_se.yml
```
Then, please use rename2.py and png2jpg_ntire.py to rename the images and change them to jpg.

# Factsheet and results on testset

1. **Download Factsheet:**
   - Navigate to [this link](https://drive.google.com/file/d/1pqLWFqFFNqlCINLmM3STV9C_Tx9gz1p7/view?usp=sharing) to download our Factsheet.
     
2. **Download results on testset:**
   - Navigate to [this link](https://drive.google.com/file/d/1vkrpr2skgrQImYA6Nv9oDqJ5bQ9iviFI/view?usp=sharing) to download our results on testset. 

# Acknowledgements

This project is built on source codes shared by [BasicSR](https://github.com/XPixelGroup/BasicSR).
