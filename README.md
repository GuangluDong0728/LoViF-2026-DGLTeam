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
   - Navigate to [this link](https://drive.google.com/drive/folders/1QYcP8mR-18SrXYNn_Tqzel03vaQIKyk5?usp=sharing) to download our weights. 

2. **Save to `experiments` Directory:**
   - Once downloaded, place the weights into the `experiments` directory.
     
# Testing

## Testing
To test our model, please open the `options/NTIRE2026/test_se.yml` file and update the paths, and just run the command:

```bash
python basicsr/test.py -opt options/NTIRE2026/test_se.yml
```

# Acknowledgements

This project is built on source codes shared by [BasicSR](https://github.com/XPixelGroup/BasicSR).
