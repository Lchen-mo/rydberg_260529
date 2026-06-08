## 这是一个课程作业程序

### 模拟中性原子量子计算中双Rb原子里德堡态布局演化

### 由于时间原因后续三维可视化没有完成

#### 一些说明：
- Qutip+pulser文件夹内是调用相关库的仿真代码；
- Phybase+3D文件夹是作者自己写的物理引擎，已知有一些物理上的bug，可正常运行，缺少3D可视化代码部分

## 环境配置

需要包含以下库：
channels:
  - conda-forge
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
  - https://repo.anaconda.com/pkgs/msys2
  
dependencies:
  - python=3.10
  - qutip
  - jupyter
  - matplotlib
  - numpy
  - scipy
  - imageio
  - ipywidgets
  - tqdm

或者具体见.yml文件