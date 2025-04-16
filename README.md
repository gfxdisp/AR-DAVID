# AR-DAVID: Augmented Reality Display Artifact Video Dataset
### [Paper](https://dongyeon93.github.io/assets/pdf/AR_DAVID.pdf) | [Project Page](https://www.cl.cam.ac.uk/research/rainbow/projects/ardavid/) | [Video](https://www.youtube.com/watch?v=DUCRmwqt-PE)

[Alexandre Chapiro](https://achapiro.github.io/)
[Dongyeon Kim](https://dongyeon93.github.io/),
[Yuta Asano](),
and [Rafał K. Mantiuk](https://www.cl.cam.ac.uk/~rkm38/)

Source code for the SIGGRAPH Asia 2024 paper titled "AR-DAVID: Augmented Reality Display Artifact Video Dataset"

## Get Started
Create anaconda environment. 
Our code has been implemented and tested on Windows.

```
conda create -n ardavid python=3.10
conda activate ardavid
```

## 1. Add the Evaluation Module
Add the cvvdp module as a Git submodule:

```bash
git submodule add https://github.com/gfxdisp/ColorVideoVDP.git
```

---

## 2. Install the ColorVideoVDP

Follow the installation instructions provided in the [ColorVideoVDP repository](https://github.com/gfxdisp/ColorVideoVDP).

Make sure to install any dependencies and build requirements mentioned there.

---

## 3. Download the AR-DAVID Dataset

Download the dataset from the following link:

🔗 [AR-DAVID Dataset](https://www.repository.cam.ac.uk/items/0b877557-9cde-49f1-a667-e88946573ee1)

### 3.1. Place or Link the Dataset

You have two options:

- **Move the dataset** to:
  ```
  datasets/AR-DAVID
  ```

- **OR**, optionally create a **symbolic link** to the dataset directory:

  On **Linux/macOS**:
  ```bash
  ln -s /path/to/AR-DAVID datasets/AR-DAVID
  ```

---

## 4. Install Additional Libraries

Install the required Python additional libraries.

---

## 5. Test

- Different optical blending methods

  ```bash
  python run_metric.py --metric=cvvdp --fusion-method={fusion_method}
  ```

  Fusion_method can be `none`,`mean`,`pinhole`,`pinhole-stereo`,`blur`,`blur-stereo`.

- Background discounting
  ```bash
  python run_metric.py --metric=cvvdp --fusion-method={fusion_method} --discount_factor={d}
  ```
  The value for d can be in a range from 0 to 1.

- For debugging
  ```bash
  python run_metric.py --metric=dm-preview --fusion-method={fusion_method} --discount_factor={d}
  ```
  This mode will save the processed videos in `dataset/AR-DAVID-{additional_method}/preview/`.

---

## 6. Plot

This is a simple MATLAB example demonstrating how to generate some of the figures in the main paper.

### 📁 Files

- `analysis/plot_across_backgrounds.m`: A script that generates the scaled quality per scene.
Colors represent the distortion type, while line styles indicate the strength of the distortion levels.

- `analysis/plot_scatter_backgrounds.m`: A script that generates a scatter plot for each optical blending method.
Colors represent background luminance levels, and markers indicate the background type.

### 🚀 How to Run

1. Open MATLAB.
2. Navigate to the folder using the command window or `cd`:

    ```matlab
    cd path/to/this/folder/analysis
    ```

3. Run the script:

    ```matlab
    plot_across_backgrounds
    plot_scatter_backgrounds
    ```

---

## Citation
```
@article{chapiro2024ar,
title={AR-DAVID: Augmented Reality Display Artifact Video Dataset},
author={Chapiro, Alexandre and Kim, Dongyeon and Asano, Yuta and Mantiuk, Rafa{\l} K},
journal={ACM Transactions on Graphics (TOG)},
volume={43},
number={6},
articleno={186},
pages={1--11},
year={2024},
publisher={ACM New York, NY, USA}
}
```

## Contact

If you have any questions, please contact

- Dongyeon Kim (dk721@cam.ac.uk)
- Rafal Mantiuk (rafal.mantiuk@cl.cam.ac.uk)