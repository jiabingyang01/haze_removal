<div align="center">

<img src="https://img.icons8.com/fluency/96/fog-day.png" width="96" alt="haze"/>
<img src="https://img.icons8.com/fluency/96/right.png" width="48" alt="arrow"/>
<img src="https://img.icons8.com/fluency/96/sunny-side-up.png" width="96" alt="clear"/>

# Haze Removal: Dark Channel Prior with AVX SIMD Optimization

**基于暗通道先验的图像/视频去雾算法 | AVX 指令集加速优化实现**

[![Language](https://img.shields.io/badge/Language-C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/)
[![SIMD](https://img.shields.io/badge/AVX_256bit-FF6F00?style=for-the-badge&logo=intel&logoColor=white)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)

</div>

---

## 📑 目录

- [🌟 项目简介](#-项目简介)
- [📐 算法原理](#-算法原理)
  - [大气散射模型](#1-大气散射模型)
  - [暗通道先验](#2-暗通道先验dark-channel-prior)
  - [大气光值估计](#3-大气光值估计)
  - [透射率估计](#4-透射率估计)
  - [导向滤波精化](#5-导向滤波精化透射率)
  - [场景辐射恢复](#6-场景辐射恢复)
- [⚡ SIMD 加速优化](#-simd-加速优化)
- [📁 项目结构](#-项目结构)
- [🔧 环境依赖](#-环境依赖)
- [🚀 编译与运行](#-编译与运行)
- [📖 使用说明](#-使用说明)
- [⚙️ 算法参数](#️-算法参数)
- [🔄 算法流程图](#-算法流程图)
- [📚 参考文献](#-参考文献)

---

## 🌟 项目简介

本项目实现了 **He Kaiming** 等人于 2009 年 CVPR 上提出的经典 **暗通道先验（Dark Channel Prior, DCP）** 单幅图像去雾算法，并在此基础上使用 **Intel AVX（Advanced Vector Extensions）256 位 SIMD 指令集** 对关键计算瓶颈进行了向量化加速优化。

项目同时支持 **静态图像** 和 **实时视频流** 的去雾处理，提供标准版本与 SIMD 加速版本两种实现，可通过参数一键切换。

### 核心特性

- **经典 DCP 算法**：完整实现暗通道提取、大气光估计、透射率计算、导向滤波、场景恢复全流程
- **AVX SIMD 加速**：对最小值滤波、导向滤波矩阵乘法、透射率计算、灰度转换等核心操作进行 256 位向量化加速
- **图像 + 视频**：支持 JPG/BMP/PNG 等图像格式与 AVI 视频流处理
- **实时性能监控**：输出每帧处理耗时、FPS 帧率等性能指标

---

## 📐 算法原理

### 1. 大气散射模型

在计算机视觉中，雾、霾等天气对图像造成的退化可由 **大气散射模型（Atmospheric Scattering Model）** 描述：

$$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$

| 符号 | 含义 |
|:---:|:---|
| $I(x)$ | 观测到的有雾图像（输入） |
| $J(x)$ | 待恢复的无雾场景辐射（目标输出） |
| $t(x)$ | 透射率（transmission map），描述光线穿过大气到达相机的比例 |
| $A$ | 全局大气光值（atmospheric light），即无穷远处的环境光强度 |

其中透射率 $t(x)$ 与场景深度 $d(x)$ 之间的关系为：

$$t(x) = e^{-\beta \cdot d(x)}$$

$\beta$ 为大气散射系数。去雾的本质就是从 $I(x)$ 中估计出 $t(x)$ 和 $A$，进而求解 $J(x)$。

### 2. 暗通道先验（Dark Channel Prior）

**核心观察**：对于户外无雾图像，绝大多数局部区域（非天空区域）中，至少存在某一个颜色通道的像素值非常低（趋近于零）。

暗通道的数学定义为：

$$J^{\text{dark}}(x) = \min_{c \in \{R,G,B\}} \left( \min_{y \in \Omega(x)} J^c(y) \right)$$

其中 $\Omega(x)$ 是以 $x$ 为中心的局部窗口。

**统计先验**：对于无雾图像 $J$，其暗通道趋近于零：

$$J^{\text{dark}}(x) \to 0$$

本项目的暗通道提取分两步实现：

```
步骤 1：逐像素取 R/G/B 三通道最小值  →  得到单通道最小值图
步骤 2：对最小值图施加最小值滤波（Min Filter）→  得到暗通道图
```

**最小值滤波窗口大小**的自适应计算：

$$\text{kernelSize} = \max\left(15,\ \max(H \times 0.01,\ W \times 0.01)\right)$$

其中 $H$、$W$ 为图像的高和宽。窗口大小取图像尺寸的 1%（至少为 15），兼顾去雾效果与边缘保持。

### 3. 大气光值估计

大气光 $A$ 的估计流程：

1. **选取候选像素**：从暗通道图中取亮度最高的前 **0.1%** 像素
2. **回溯原图定位**：利用这些像素在暗通道中的坐标，回到原始有雾图像中查找对应位置
3. **选择最亮点**：在候选像素集合中，选取 RGB 均值最大的那个像素的 RGB 值作为大气光 $A = (A_R, A_G, A_B)$

$$A = \arg\max_{(x,y) \in \text{Top}_{0.1\%}} \frac{I_R(x,y) + I_G(x,y) + I_B(x,y)}{3}$$

### 4. 透射率估计

将大气散射模型两边除以大气光 $A$，得到归一化形式：

$$\frac{I^c(x)}{A^c} = t(x) \cdot \frac{J^c(x)}{A^c} + (1 - t(x))$$

对归一化后的图像取暗通道：

$$\min_c\left(\min_{y \in \Omega(x)} \frac{I^c(y)}{A^c}\right) = t(x) \cdot \underbrace{\min_c\left(\min_{y \in \Omega(x)} \frac{J^c(y)}{A^c}\right)}_{\to\ 0\ \text{（暗通道先验）}} + (1 - t(x))$$

由暗通道先验 $J^{\text{dark}} \to 0$，化简得到透射率估计：

$$\tilde{t}(x) = 1 - \omega \cdot \min_c\left(\min_{y \in \Omega(x)} \frac{I^c(y)}{A^c}\right)$$

其中 $\omega = 0.95$ 是一个保留因子，保留少量雾气以增强画面的自然纵深感。

### 5. 导向滤波精化透射率

直接估计的透射率 $\tilde{t}(x)$ 在边缘区域存在块效应（block artifacts），需要进行边缘保持的平滑滤波。本项目使用 **导向滤波（Guided Filter）** 代替原论文中的 Soft Matting，以灰度图作为引导图像：

**导向滤波的核心公式**：

假设导向图像为 $I$，输入为 $p$，输出为 $q$，则在每个局部窗口 $\omega_k$ 中，输出是导向图的线性变换：

$$q_i = a_k \cdot I_i + b_k, \quad \forall i \in \omega_k$$

线性系数通过最小化重建误差求得：

$$a_k = \frac{\frac{1}{|\omega|}\sum_{i \in \omega_k} I_i p_i - \mu_k \bar{p}_k}{\sigma_k^2 + \epsilon} = \frac{\text{Cov}(I, p)}{\text{Var}(I) + \epsilon}$$

$$b_k = \bar{p}_k - a_k \cdot \mu_k$$

最终输出为所有包含像素 $i$ 的窗口系数的均值：

$$q_i = \bar{a}_i \cdot I_i + \bar{b}_i$$

| 参数 | 值 | 说明 |
|:---:|:---:|:---|
| 引导图 | 灰度图 | 从 BGR 转换的单通道灰度图 |
| 滤波半径 $r$ | $6 \times \text{kernelSize}$ | 较大的窗口确保充分平滑 |
| 正则化参数 $\epsilon$ | $0.001$ | 控制平滑程度，值越小越贴合引导图边缘 |

### 6. 场景辐射恢复

利用估计的 $A$ 和精化后的 $t(x)$，由大气散射模型反解无雾图像：

$$J(x) = \frac{I(x) - A}{\max(t(x),\ t_0)} + A$$

其中 $t_0 = 0.5$ 是透射率下限阈值，防止 $t(x)$ 过小导致噪声放大。

此外，恢复后还叠加了一个微小的亮度补偿 $\Delta = \frac{10}{255}$，以改善整体视觉观感。

---

## ⚡ SIMD 加速优化

本项目在标准实现的基础上，对以下四个计算热点使用 **Intel AVX 256-bit 指令集**进行了并行加速，每次处理 **8 个 float 元素**：

### 优化模块一览

| 模块 | 标准函数 | SIMD 函数 | 核心 AVX 指令 |
|:---|:---|:---|:---|
| 最小值滤波 | `minFilter()` | `minFilter_SIMD()` | `_mm256_min_ps` |
| 矩阵逐元素乘法 | `cv::multiply()` | `multiplyAVX()` | `_mm256_mul_ps` |
| 导向滤波 | `guidedFilter()` | `guidedFilter_SIMD()` | 调用 `multiplyAVX` |
| 透射率计算 | `getTransmission_dark()` | `getTransmission_dark_SIMD()` | `_mm256_sub_ps`, `_mm256_mul_ps` |
| 灰度归一化 | 逐元素循环 | AVX 批量转换 | `_mm256_cvtepu8_epi32`, `_mm256_div_ps` |

### 最小值滤波 SIMD 优化示例

标准版使用 `cv::minMaxLoc()` 逐像素提取 ROI 最小值；SIMD 版将卷积核遍历向量化：

```cpp
// 一次加载 8 个 float，取 SIMD 寄存器级并行最小值
__m256 minValues = _mm256_set1_ps(FLT_MAX);
for (int kr = -radius; kr <= radius; kr++) {
    for (int kc = -radius; kc <= radius; kc++) {
        __m256 roiValues = _mm256_loadu_ps(ptr + offset);
        minValues = _mm256_min_ps(minValues, roiValues);
    }
}
```

对于不满 8 元素对齐的尾部数据，自动回退到标量处理，确保正确性。

---

## 📁 项目结构

```
haze_removal/
├── README.md                              # 项目说明文档
├── src/
│   ├── deHazeByDarkChannelPrior.h         # 头文件：函数声明与依赖包含
│   ├── deHazeByDarkChannelPrior.cpp       # 核心算法：标准版 + SIMD 版（~1016 行）
│   ├── demo.cpp                           # 入口程序：图像/视频测试与输出
│   └── bin/
│       ├── deHazeByDarkChannelPrior.exe   # 编译后的算法模块
│       └── demo.exe                       # 编译后的可执行演示程序
├── input/
│   ├── images/                            # 测试图像（8 张，JPG/BMP 格式）
│   │   ├── 1.jpg, 2.jpg
│   │   └── 3.bmp ~ 8.bmp
│   └── videos/                            # 测试视频（2 个，AVI 格式）
│       ├── cross.avi
│       └── riverside.avi
└── output/
    ├── images/                            # 图像去雾输出目录
    └── videos/                            # 视频去雾输出目录
```

### 源码模块说明

| 文件 | 核心函数 | 说明 |
|:---|:---|:---|
| `deHazeByDarkChannelPrior.cpp` | `deHazeByDarkChannelPrior()` | 标准去雾全流程入口 |
| | `deHazeByDarkChannelPrior_SIMD()` | SIMD 加速去雾全流程入口 |
| | `minFilter()` / `minFilter_SIMD()` | 最小值滤波（暗通道提取核心） |
| | `guidedFilter()` / `guidedFilter_SIMD()` | 导向滤波（透射率精化） |
| | `getTransmission_dark()` / `..._SIMD()` | 透射率估计 + 导向滤波调用 |
| | `recover()` | 场景辐射恢复 |
| | `multiplyAVX()` | AVX 矩阵逐元素乘法 |
| `demo.cpp` | `mat_testOnImg()` | 对 Mat 矩阵进行去雾测试并显示 |
| | `file_testOnImg()` | 从文件加载图像并测试 |
| | `writeImg()` | 去雾并将结果写入 PNG 文件 |
| | `testOnMedia()` | 实时视频去雾并显示帧率 |
| | `writeMedia()` | 视频去雾并写入 AVI 文件 |

---

## 🔧 环境依赖

| 依赖项 | 要求 |
|:---|:---|
| **C++ 标准** | C++11 或更高 |
| **OpenCV** | 2.0+（推荐 3.x / 4.x） |
| **CPU 指令集** | 支持 AVX 的 x86-64 处理器（Intel Sandy Bridge / AMD Bulldozer 及以后） |
| **编译器** | MSVC（推荐）/ GCC / Clang，需开启 AVX 支持 |
| **操作系统** | Windows（已提供预编译 .exe），Linux/macOS 可自行编译 |

**OpenCV 模块依赖**：`core`, `imgproc`, `highgui`, `imgcodecs`, `videoio`

---

## 🚀 编译与运行

### 编译

以 MSVC 为例：

```bash
cl /O2 /arch:AVX2 /EHsc /I<opencv_include_path> src/demo.cpp src/deHazeByDarkChannelPrior.cpp /link /LIBPATH:<opencv_lib_path> opencv_world4xx.lib
```

以 g++ 为例：

```bash
g++ -O2 -mavx2 -std=c++11 src/demo.cpp src/deHazeByDarkChannelPrior.cpp \
    `pkg-config --cflags --libs opencv4` -o demo
```

### 运行

直接执行预编译程序或自行编译的可执行文件：

```bash
./demo
```

运行模式通过 `demo.cpp` 中的 `main()` 函数参数控制（修改源码后重新编译）：

```cpp
int type = 1;       // 0 = 图像模式, 1 = 视频模式
int out = 0;        // 0 = 窗口显示, 1 = 写入文件
int use_simd = 1;   // 0 = 标准版本, 1 = SIMD 加速版本
String name = "cross";  // 输入文件名（不含扩展名）
```

---

## 📖 使用说明

### 图像去雾

将待处理图像放入 `input/images/` 目录，修改 `demo.cpp` 中的配置：

```cpp
int type = 0;                // 切换为图像模式
cv::String name = "1";       // 文件名（对应 input/images/1.jpg）
```

### 视频去雾

将待处理视频放入 `input/videos/` 目录，修改 `demo.cpp` 中的配置：

```cpp
int type = 1;                // 切换为视频模式
cv::String name = "cross";   // 文件名（对应 input/videos/cross.avi）
```

### 切换标准/SIMD 模式

```cpp
int use_simd = 1;  // 1 = AVX SIMD 加速, 0 = 标准实现
```

---

## ⚙️ 算法参数

| 参数 | 符号 | 默认值 | 说明 |
|:---|:---:|:---:|:---|
| 保留因子 | $\omega$ | `0.95` | 控制去雾强度，保留 5% 远景雾气以增强纵深感 |
| 透射率下限 | $t_0$ | `0.5` | 防止透射率过低导致恢复图像噪声放大 |
| 导向滤波正则化 | $\epsilon$ | `0.001` | 控制导向滤波的平滑-保边权衡 |
| 导向滤波半径 | $r$ | $6 \times \text{kernelSize}$ | 滤波窗口大小 |
| 大气光候选比例 | — | `0.1%` | 从暗通道中选取最亮的 0.1% 像素作为候选 |
| 亮度补偿 | $\Delta$ | $10/255$ | 恢复后的全局亮度微调 |
| 滤波核大小 | — | $\max(15, 1\%\text{图像尺寸})$ | 自适应窗口，兼顾效果与边缘保持 |

---

## 🔄 算法流程图

```
┌─────────────────────────────┐
│       输入有雾图像 I(x)       │
│       (BGR, uint8)          │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  转换为 32-bit 浮点 (0~1)    │
│  I_float = I / 255.0        │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│    提取暗通道 D(x)           │
│  = min_c(min_Ω I^c(y))     │
│  ① 逐像素 RGB 取最小值       │
│  ② 局部窗口最小值滤波         │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   估计大气光 A               │
│  取暗通道前 0.1% 最亮像素     │
│  在原图中选 RGB 均值最大者     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   归一化暗通道 (I/A)         │
│  并施加最小值滤波             │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   计算透射率                 │
│  t(x) = 1 - ω · D_norm(x)  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   导向滤波精化透射率          │
│  引导图 = 灰度图             │
│  r = 6·kernelSize, ε=0.001  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   恢复无雾场景               │
│  J = (I-A)/max(t, 0.5) + A  │
│  + 亮度补偿 Δ                │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   转回 uint8 (×255)         │
│   输出去雾图像 J(x)          │
└─────────────────────────────┘
```

---

## 📚 参考文献

1. **He K, Sun J, Tang X.** *Single Image Haze Removal Using Dark Channel Prior.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2011, 33(12): 2341-2353. [[PDF]](https://kaiminghe.com/publications/pami10dehaze.pdf)

2. **He K, Sun J, Tang X.** *Guided Image Filtering.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2013, 35(6): 1397-1409. [[PDF]](https://kaiminghe.com/publications/pami12guidedfilter.pdf)

3. **Intel Intrinsics Guide** — AVX/AVX2 指令集参考. [[Link]](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)

---

<div align="center">

*基于 He Kaiming 暗通道先验理论，C++ & OpenCV & AVX SIMD 实现*

</div>
