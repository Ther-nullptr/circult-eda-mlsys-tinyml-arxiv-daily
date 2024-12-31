[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

## Updated on 2024.12.31
> Usage instructions: [here](./docs/README.md#usage)

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href=#quantization>Quantization</a></li>
    <li><a href=#pruning>Pruning</a></li>
    <li><a href=#hardware-software-co-design>Hardware-Software Co-Design</a></li>
    <li><a href=#tinyml>TinyML</a></li>
    <li><a href=#domain-specific-accelerator>Domain Specific Accelerator</a></li>
    <li><a href=#low-rank-adaptation>Low-Rank Adaptation</a></li>
    <li><a href=#model-compression>Model Compression</a></li>
  </ol>
</details>

## Quantization

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-12-30**|**Improving Acoustic Scene Classification in Low-Resource Conditions**|Zhi Chen et.al.|[2412.20722](http://arxiv.org/abs/2412.20722)|null|
|**2024-12-29**|**PTQ4VM: Post-Training Quantization for Visual Mamba**|Younghyun Cho et.al.|[2412.20386](http://arxiv.org/abs/2412.20386)|null|
|**2024-12-28**|**IMSSA: Deploying modern state-space models on memristive in-memory compute hardware**|Sebastian Siegel et.al.|[2412.20215](http://arxiv.org/abs/2412.20215)|null|
|**2024-12-27**|**Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales**|Shuokai Pan et.al.|[2412.19867](http://arxiv.org/abs/2412.19867)|null|
|**2024-12-27**|**MBQ: Modality-Balanced Quantization for Large Vision-Language Models**|Shiyao Li et.al.|[2412.19509](http://arxiv.org/abs/2412.19509)|**[link](https://github.com/thu-nics/mbq)**|
|**2024-12-24**|**Unified Stochastic Framework for Neural Network Quantization and Pruning**|Haoyu Zhang et.al.|[2412.18184](http://arxiv.org/abs/2412.18184)|null|
|**2024-12-21**|**TCAQ-DM: Timestep-Channel Adaptive Quantization for Diffusion Models**|Haocheng Huang et.al.|[2412.16700](http://arxiv.org/abs/2412.16700)|null|
|**2024-12-20**|**Improving Quantization-aware Training of Low-Precision Network via Block Replacement on Full-Precision Counterpart**|Chengting Yu et.al.|[2412.15846](http://arxiv.org/abs/2412.15846)|null|
|**2024-12-19**|**Progressive Fine-to-Coarse Reconstruction for Accurate Low-Bit Post-Training Quantization in Vision Transformers**|Rui Ding et.al.|[2412.14633](http://arxiv.org/abs/2412.14633)|null|
|**2024-12-19**|**Qua $^2$ SeDiMo: Quantifiable Quantization Sensitivity of Diffusion Models**|Keith G. Mills et.al.|[2412.14628](http://arxiv.org/abs/2412.14628)|null|
|**2024-12-18**|**ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals**|Utkarsh Saxena et.al.|[2412.14363](http://arxiv.org/abs/2412.14363)|**[link](https://github.com/utkarsh-dmx/project-resq)**|
|**2024-12-15**|**Efficient Quantization-Aware Training on Segment Anything Model in Medical Images and Its Deployment**|Haisheng Lu et.al.|[2412.11186](http://arxiv.org/abs/2412.11186)|**[link](https://github.com/avc2-uestc/qmedsam)**|
|**2024-12-13**|**TTAQ: Towards Stable Post-training Quantization in Continuous Domain Adaptation**|Junrui Xiao et.al.|[2412.09899](http://arxiv.org/abs/2412.09899)|null|
|**2024-12-12**|**CRVQ: Channel-relaxed Vector Quantization for Extreme Compression of LLMs**|Yuzhuang Xu et.al.|[2412.09282](http://arxiv.org/abs/2412.09282)|null|
|**2024-12-10**|**Post-Training Non-Uniform Quantization for Convolutional Neural Networks**|Ahmed Luqman et.al.|[2412.07391](http://arxiv.org/abs/2412.07391)|null|
|**2024-12-09**|**FP=xINT:A Low-Bit Series Expansion Algorithm for Post-Training Quantization**|Boyang Zhang et.al.|[2412.06865](http://arxiv.org/abs/2412.06865)|null|
|**2024-12-09**|**Efficiency Meets Fidelity: A Novel Quantization Framework for Stable Diffusion**|Shuaiting Li et.al.|[2412.06661](http://arxiv.org/abs/2412.06661)|null|
|**2024-12-07**|**GAQAT: gradient-adaptive quantization-aware training for domain generalization**|Jiacheng Jiang et.al.|[2412.05551](http://arxiv.org/abs/2412.05551)|null|
|**2024-12-07**|**SKIM: Any-bit Quantization Pushing The Limits of Post-Training Quantization**|Runsheng Bai et.al.|[2412.04180](http://arxiv.org/abs/2412.04180)|null|
|**2024-12-05**|**Quantized and Interpretable Learning Scheme for Deep Neural Networks in Classification Task**|Alireza Maleki et.al.|[2412.03915](http://arxiv.org/abs/2412.03915)|null|
|**2024-12-03**|**CPTQuant - A Novel Mixed Precision Post-Training Quantization Techniques for Large Language Models**|Amitash Nanda et.al.|[2412.03599](http://arxiv.org/abs/2412.03599)|null|
|**2024-11-26**|**Rapid Deployment of Domain-specific Hyperspectral Image Processors with Application to Autonomous Driving**|Jon Gutiérrez-Zaballa et.al.|[2411.17543](http://arxiv.org/abs/2411.17543)|null|
|**2024-12-03**|**PassionSR: Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution**|Libo Zhu et.al.|[2411.17106](http://arxiv.org/abs/2411.17106)|**[link](https://github.com/libozhu03/passionsr)**|
|**2024-11-23**|**freePruner: A Training-free Approach for Large Multimodal Model Acceleration**|Bingxin Xu et.al.|[2411.15446](http://arxiv.org/abs/2411.15446)|null|
|**2024-11-22**|**FLARE: FP-Less PTQ and Low-ENOB ADC Based AMS-PiM for Error-Resilient, Fast, and Efficient Transformer Acceleration**|Donghyeon Yi et.al.|[2411.14733](http://arxiv.org/abs/2411.14733)|null|
|**2024-11-17**|**EfQAT: An Efficient Framework for Quantization-Aware Training**|Saleh Ashkboos et.al.|[2411.11038](http://arxiv.org/abs/2411.11038)|null|
|**2024-11-12**|**ASER: Activation Smoothing and Error Reconstruction for Large Language Model Quantization**|Weibo Zhao et.al.|[2411.07762](http://arxiv.org/abs/2411.07762)|null|
|**2024-11-09**|**Optimizing Large Language Models through Quantization: A Comparative Analysis of PTQ and QAT Techniques**|Jahid Hasan et.al.|[2411.06084](http://arxiv.org/abs/2411.06084)|null|
|**2024-11-08**|**SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**|Muyang Li et.al.|[2411.05007](http://arxiv.org/abs/2411.05007)|**[link](https://github.com/mit-han-lab/deepcompressor)**|
|**2024-11-30**|**Scaling Laws for Precision**|Tanishq Kumar et.al.|[2411.04330](http://arxiv.org/abs/2411.04330)|null|
|**2024-11-06**|**Interactions Across Blocks in Post-Training Quantization of Large Language Models**|Khasmamad Shabanovi et.al.|[2411.03934](http://arxiv.org/abs/2411.03934)|null|
|**2024-11-06**|**An Edge Computing-Based Solution for Real-Time Leaf Disease Classification using Thermal Imaging**|Públio Elon Correa da Silva et.al.|[2411.03835](http://arxiv.org/abs/2411.03835)|**[link](https://github.com/publioelon/leaf-diseases-classification)**|
|**2024-11-06**|**TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture**|Jiajun Wu et.al.|[2411.03697](http://arxiv.org/abs/2411.03697)|null|
|**2024-10-29**|**Data Generation for Hardware-Friendly Post-Training Quantization**|Lior Dikstein et.al.|[2410.22110](http://arxiv.org/abs/2410.22110)|**[link](https://github.com/sony/model_optimization)**|
|**2024-10-30**|**IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models**|Hang Guo et.al.|[2410.21759](http://arxiv.org/abs/2410.21759)|**[link](https://github.com/csguoh/intlora)**|
|**2024-10-26**|**DQRM: Deep Quantized Recommendation Models**|Yang Zhou et.al.|[2410.20046](http://arxiv.org/abs/2410.20046)|**[link](https://github.com/YangZhou08/Deep_Quantized_Recommendation_Model_DQRM)**|
|**2024-10-14**|**Real-Time Stress Detection via Photoplethysmogram Signals: Implementation of a Combined Continuous Wavelet Transform and Convolutional Neural Network on Resource-Constrained Microcontrollers**|Yasin Hasanpoor et.al.|[2410.19776](http://arxiv.org/abs/2410.19776)|null|
|**2024-10-24**|**TesseraQ: Ultra Low-Bit LLM Post-Training Quantization with Block Reconstruction**|Yuhang Li et.al.|[2410.19103](http://arxiv.org/abs/2410.19103)|null|
|**2024-10-18**|**Understanding the difficulty of low-precision post-training quantization of large language models**|Zifei Xu et.al.|[2410.14570](http://arxiv.org/abs/2410.14570)|null|
|**2024-10-17**|**Quamba: A Post-Training Quantization Recipe for Selective State Space Models**|Hung-Yueh Chiang et.al.|[2410.13229](http://arxiv.org/abs/2410.13229)|**[link](https://github.com/enyac-group/quamba)**|
|**2024-10-17**|**Scaling laws for post-training quantized large language models**|Zifei Xu et.al.|[2410.12119](http://arxiv.org/abs/2410.12119)|null|
|**2024-10-15**|**Error Diffusion: Post Training Quantization with Block-Scaled Number Formats for Neural Networks**|Alireza Khodamoradi et.al.|[2410.11203](http://arxiv.org/abs/2410.11203)|**[link](https://github.com/rocm/tensorcast)**|
|**2024-10-06**|**Continuous Approximations for Improving Quantization Aware Training of LLMs**|He Li et.al.|[2410.10849](http://arxiv.org/abs/2410.10849)|null|
|**2024-10-12**|**SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs**|Mohammad Mozaffari et.al.|[2410.09615](http://arxiv.org/abs/2410.09615)|**[link](https://github.com/mohammad-mozaffari/slim)**|
|**2024-10-12**|**FlatQuant: Flatness Matters for LLM Quantization**|Yuxuan Sun et.al.|[2410.09426](http://arxiv.org/abs/2410.09426)|**[link](https://github.com/ruikangliu/flatquant)**|
|**2024-10-10**|**Q-VLM: Post-training Quantization for Large Vision-Language Models**|Changyuan Wang et.al.|[2410.08119](http://arxiv.org/abs/2410.08119)|**[link](https://github.com/changyuanwang17/qvlm)**|
|**2024-10-10**|**Post-Training Quantization in Brain-Computer Interfaces based on Event-Related Potential Detection**|Hubert Cecotti et.al.|[2410.07920](http://arxiv.org/abs/2410.07920)|null|
|**2024-10-10**|**CrossQuant: A Post-Training Quantization Method with Smaller Quantization Kernel for Precise Large Language Model Compression**|Wenyuan Liu et.al.|[2410.07505](http://arxiv.org/abs/2410.07505)|null|
|**2024-10-09**|**Scaling Laws for Mixed quantization in Large Language Models**|Zeyu Cao et.al.|[2410.06722](http://arxiv.org/abs/2410.06722)|null|
|**2024-10-08**|**QERA: an Analytical Framework for Quantization Error Reconstruction**|Cheng Zhang et.al.|[2410.06040](http://arxiv.org/abs/2410.06040)|null|
|**2024-10-08**|**QT-DoG: Quantization-aware Training for Domain Generalization**|Saqib Javed et.al.|[2410.06020](http://arxiv.org/abs/2410.06020)|**[link](https://github.com/saqibjaved1/QT-DoG)**|
|**2024-10-10**|**ARB-LLM: Alternating Refined Binarizations for Large Language Models**|Zhiteng Li et.al.|[2410.03129](http://arxiv.org/abs/2410.03129)|**[link](https://github.com/zhitengli/arb-llm)**|
|**2024-10-03**|**Lightweight Diffusion Models for Resource-Constrained Semantic Communication**|Giovanni Pignata et.al.|[2410.02491](http://arxiv.org/abs/2410.02491)|**[link](https://github.com/ispamm/q-gesco)**|
|**2024-10-01**|**Compressing Recurrent Neural Networks for FPGA-accelerated Implementation in Fluorescence Lifetime Imaging**|Ismail Erbas et.al.|[2410.00948](http://arxiv.org/abs/2410.00948)|null|
|**2024-09-30**|**Constraint Guided Model Quantization of Neural Networks**|Quinten Van Baelen et.al.|[2409.20138](http://arxiv.org/abs/2409.20138)|null|
|**2024-09-26**|**P4Q: Learning to Prompt for Quantization in Visual-language Models**|Huixin Sun et.al.|[2409.17634](http://arxiv.org/abs/2409.17634)|null|
|**2024-09-25**|**Accumulator-Aware Post-Training Quantization**|Ian Colbert et.al.|[2409.17092](http://arxiv.org/abs/2409.17092)|null|
|**2024-09-25**|**VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models**|Yifei Liu et.al.|[2409.17066](http://arxiv.org/abs/2409.17066)|**[link](https://github.com/microsoft/vptq)**|
|**2024-09-25**|**PTQ4RIS: Post-Training Quantization for Referring Image Segmentation**|Xiaoyan Jiang et.al.|[2409.17020](http://arxiv.org/abs/2409.17020)|**[link](https://github.com/gugu511yy/ptq4ris)**|
|**2024-09-26**|**INT-FlashAttention: Enabling Flash Attention for INT8 Quantization**|Shimao Chen et.al.|[2409.16997](http://arxiv.org/abs/2409.16997)|**[link](https://github.com/int-flashattention2024/int-flashattention)**|
|**2024-09-20**|**PTQ4ADM: Post-Training Quantization for Efficient Text Conditional Audio Diffusion Models**|Jayneel Vora et.al.|[2409.13894](http://arxiv.org/abs/2409.13894)|null|
|**2024-09-18**|**Art and Science of Quantizing Large-Scale Models: A Comprehensive Overview**|Yanshu Wang et.al.|[2409.11650](http://arxiv.org/abs/2409.11650)|null|
|**2024-09-12**|**LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs**|Han Xu et.al.|[2409.11424](http://arxiv.org/abs/2409.11424)|null|
|**2024-09-12**|**DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing**|Zhenyuan Dong et.al.|[2409.07756](http://arxiv.org/abs/2409.07756)|**[link](https://github.com/DZY122/DiTAS)**|
|**2024-08-31**|**Accurate Compression of Text-to-Image Diffusion Models via Vector Quantization**|Vage Egiazarian et.al.|[2409.00492](http://arxiv.org/abs/2409.00492)|null|
|**2024-08-29**|**A machine learning approach for computing solar flare locations in X-rays on-board Solar Orbiter/STIX**|Paolo Massa et.al.|[2408.16642](http://arxiv.org/abs/2408.16642)|**[link](https://github.com/paolomassa/STX_CFL_NN)**|
|**2024-08-29**|**On-device AI: Quantization-aware Training of Transformers in Time-Series**|Tianheng Ling et.al.|[2408.16495](http://arxiv.org/abs/2408.16495)|null|
|**2024-08-27**|**The Uniqueness of LLaMA3-70B with Per-Channel Quantization: An Empirical Study**|Minghai Qin et.al.|[2408.15301](http://arxiv.org/abs/2408.15301)|null|
|**2024-08-25**|**MobileQuant: Mobile-friendly Quantization for On-device Language Models**|Fuwen Tan et.al.|[2408.13933](http://arxiv.org/abs/2408.13933)|**[link](https://github.com/saic-fi/mobilequant)**|
|**2024-08-25**|**Infrared Domain Adaptation with Zero-Shot Quantization**|Burak Sevsay et.al.|[2408.13925](http://arxiv.org/abs/2408.13925)|null|
|**2024-08-23**|**ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration for Large Language Models**|Chao Zeng et.al.|[2408.08554](http://arxiv.org/abs/2408.08554)|**[link](https://github.com/bytedance/abq-llm)**|
|**2024-08-14**|**Analog Spiking Neuron in CMOS 28 nm Towards Large-Scale Neuromorphic Processors**|Marwan Besrour et.al.|[2408.07734](http://arxiv.org/abs/2408.07734)|null|
|**2024-08-13**|**Low-Bitwidth Floating Point Quantization for Efficient High-Quality Diffusion Models**|Cheng Chen et.al.|[2408.06995](http://arxiv.org/abs/2408.06995)|null|
|**2024-08-11**|**RTF-Q: Unsupervised domain adaptation based retraining-free quantization network**|Nanyang Du et.al.|[2408.05752](http://arxiv.org/abs/2408.05752)|null|
|**2024-08-16**|**DopQ-ViT: Towards Distribution-Friendly and Outlier-Aware Post-Training Quantization for Vision Transformers**|Lianwei Yang et.al.|[2408.03291](http://arxiv.org/abs/2408.03291)|null|
|**2024-08-05**|**HQOD: Harmonious Quantization for Object Detection**|Long Huang et.al.|[2408.02561](http://arxiv.org/abs/2408.02561)|**[link](https://github.com/Menace-Dragon/VP-QOD)**|
|**2024-08-01**|**Reclaiming Residual Knowledge: A Novel Paradigm to Low-Bit Quantization**|Róisín Luo et.al.|[2408.00923](http://arxiv.org/abs/2408.00923)|null|
|**2024-08-07**|**Temporal Feature Matters: A Framework for Diffusion Model Quantization**|Yushi Huang et.al.|[2407.19547](http://arxiv.org/abs/2407.19547)|null|
|**2024-07-25**|**Unlocking Tokens as Data Points for Generalization Bounds on Larger Language Models**|Sanae Lotfi et.al.|[2407.18158](http://arxiv.org/abs/2407.18158)|null|
|**2024-07-27**|**MetaAug: Meta-Data Augmentation for Post-Training Quantization**|Cuong Pham et.al.|[2407.14726](http://arxiv.org/abs/2407.14726)|**[link](https://github.com/cuong-pv/MetaAug-PTQ)**|
|**2024-07-17**|**AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer**|Zhuguanyu Wu et.al.|[2407.12951](http://arxiv.org/abs/2407.12951)|**[link](https://github.com/GoatWu/AdaLog)**|
|**2024-07-17**|**Mamba-PTQ: Outlier Channels in Recurrent Large Language Models**|Alessandro Pierro et.al.|[2407.12397](http://arxiv.org/abs/2407.12397)|null|
|**2024-07-17**|**StoX-Net: Stochastic Processing of Partial Sums for Efficient In-Memory Computing DNN Accelerators**|Ethan G Rogers et.al.|[2407.12378](http://arxiv.org/abs/2407.12378)|null|
|**2024-07-17**|**Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language Models**|Ayush Kaushal et.al.|[2407.12327](http://arxiv.org/abs/2407.12327)|**[link](https://github.com/nolanoorg/spectrasuite)**|
|**2024-07-17**|**QVD: Post-training Quantization for Video Diffusion Models**|Shilong Tian et.al.|[2407.11585](http://arxiv.org/abs/2407.11585)|null|
|**2024-07-16**|**LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices**|Jung Hyun Lee et.al.|[2407.11534](http://arxiv.org/abs/2407.11534)|**[link](https://github.com/onliwad101/flexround_lrq)**|
|**2024-07-11**|**Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients**|Zhenyu Zhang et.al.|[2407.08296](http://arxiv.org/abs/2407.08296)|**[link](https://github.com/VITA-Group/Q-GaLore)**|
|**2024-07-10**|**RoLoRA: Fine-tuning Rotated Outlier-free LLMs for Effective Weight-Activation Quantization**|Xijie Huang et.al.|[2407.08044](http://arxiv.org/abs/2407.08044)|**[link](https://github.com/huangowen/rolora)**|

<p align=right>(<a href=#updated-on-20241231>back to top</a>)</p>

## Pruning

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-12-24**|**SlimGPT: Layer-wise Structured Pruning for Large Language Models**|Gui Ling et.al.|[2412.18110](http://arxiv.org/abs/2412.18110)|null|
|**2024-12-23**|**GQSA: Group Quantization and Sparsity for Accelerating Large Language Model Inference**|Chao Zeng et.al.|[2412.17560](http://arxiv.org/abs/2412.17560)|null|
|**2024-12-28**|**Lillama: Large Language Models Compression via Low-Rank Feature Distillation**|Yaya Sy et.al.|[2412.16719](http://arxiv.org/abs/2412.16719)|null|
|**2024-12-21**|**V"Mean"ba: Visual State Space Models only need 1 hidden dimension**|Tien-Yu Chi et.al.|[2412.16602](http://arxiv.org/abs/2412.16602)|null|
|**2024-12-20**|**Less is More: Towards Green Code Large Language Models via Unified Structural Pruning**|Guang Yang et.al.|[2412.15921](http://arxiv.org/abs/2412.15921)|null|
|**2024-12-20**|**All-in-One Tuning and Structural Pruning for Domain-Specific LLMs**|Lei Lu et.al.|[2412.14426](http://arxiv.org/abs/2412.14426)|null|
|**2024-12-17**|**Learning Coarse-to-Fine Pruning of Graph Convolutional Networks for Skeleton-based Recognition**|Hichem Sahbi et.al.|[2412.12887](http://arxiv.org/abs/2412.12887)|null|
|**2024-12-17**|**A Comparative Study of Pruning Methods in Transformer-based Time Series Forecasting**|Nicholas Kiefer et.al.|[2412.12883](http://arxiv.org/abs/2412.12883)|null|
|**2024-12-17**|**Structural Pruning via Spatial-aware Information Redundancy for Semantic Segmentation**|Dongyue Wu et.al.|[2412.12672](http://arxiv.org/abs/2412.12672)|**[link](https://github.com/dywu98/SIRFP)**|
|**2024-12-19**|**RemoteTrimmer: Adaptive Structural Pruning for Remote Sensing Image Classification**|Guangwenjie Zou et.al.|[2412.12603](http://arxiv.org/abs/2412.12603)|**[link](https://github.com/1e12Leon/RemoteTrimmer)**|
|**2024-12-16**|**Designing Semi-Structured Pruning of Graph Convolutional Networks for Skeleton-based Recognition**|Hichem Sahbi et.al.|[2412.11813](http://arxiv.org/abs/2412.11813)|null|
|**2024-12-16**|**QPruner: Probabilistic Decision Quantization for Structured Pruning in Large Language Models**|Changhai Zhou et.al.|[2412.11629](http://arxiv.org/abs/2412.11629)|null|
|**2024-12-09**|**LLM-BIP: Structured Pruning for Large Language Models with Block-Wise Forward Importance Propagation**|Haihang Wu et.al.|[2412.06419](http://arxiv.org/abs/2412.06419)|null|
|**2024-12-03**|**Effortless Efficiency: Low-Cost Pruning of Diffusion Models**|Yang Zhang et.al.|[2412.02852](http://arxiv.org/abs/2412.02852)|null|
|**2024-11-25**|**Deep Convolutional Neural Networks Structured Pruning via Gravity Regularization**|Abdesselam Ferdi et.al.|[2411.16901](http://arxiv.org/abs/2411.16901)|null|
|**2024-11-21**|**FuseGPT: Learnable Layers Fusion of Generative Pre-trained Transformers**|Zehua Pei et.al.|[2411.14507](http://arxiv.org/abs/2411.14507)|null|
|**2024-11-21**|**Layer Pruning with Consensus: A Triple-Win Solution**|Leandro Giusti Mugnaini et.al.|[2411.14345](http://arxiv.org/abs/2411.14345)|**[link](https://github.com/carolinatavaresduarte/consensus-layer-pruning)**|
|**2024-11-21**|**DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization**|Hexuan Deng et.al.|[2411.14055](http://arxiv.org/abs/2411.14055)|**[link](https://github.com/hexuandeng/drpruning)**|
|**2024-11-19**|**FGP: Feature-Gradient-Prune for Efficient Convolutional Layer Pruning**|Qingsong Lv et.al.|[2411.12781](http://arxiv.org/abs/2411.12781)|**[link](https://github.com/fgp-code/fgp)**|
|**2024-11-17**|**Electrostatic Force Regularization for Neural Structured Pruning**|Abdesselam Ferdi et.al.|[2411.11079](http://arxiv.org/abs/2411.11079)|null|
|**2024-11-15**|**Systolic Arrays and Structured Pruning Co-design for Efficient Transformers in Edge Systems**|Pedro Palacios et.al.|[2411.10285](http://arxiv.org/abs/2411.10285)|null|
|**2024-12-16**|**P $^2$ Law: Scaling Law for Post-Training After Model Pruning**|Xiaodong Chen et.al.|[2411.10272](http://arxiv.org/abs/2411.10272)|null|
|**2024-11-10**|**RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration**|Boyao Wang et.al.|[2411.06463](http://arxiv.org/abs/2411.06463)|**[link](https://github.com/beryex/rlpruner-cnn)**|
|**2024-11-05**|**Layer-Adaptive State Pruning for Deep State Space Models**|Minseon Gwak et.al.|[2411.02824](http://arxiv.org/abs/2411.02824)|**[link](https://github.com/msgwak/last)**|
|**2024-11-04**|**Automatic Structured Pruning for Efficient Architecture in Federated Learning**|Thai Vu Nguyen et.al.|[2411.01759](http://arxiv.org/abs/2411.01759)|**[link](https://github.com/NguyenThaiVu/prune_fl_project)**|
|**2024-10-31**|**Mutual Information Preserving Neural Network Pruning**|Charles Westphal et.al.|[2411.00147](http://arxiv.org/abs/2411.00147)|null|
|**2024-10-24**|**Tailored-LLaMA: Optimizing Few-Shot Learning in Pruned LLaMA Models with Task-Specific Prompts**|Danyal Aftab et.al.|[2410.19185](http://arxiv.org/abs/2410.19185)|null|
|**2024-10-18**|**EvoPress: Towards Optimal Dynamic Model Compression via Evolutionary Search**|Oliver Sieberling et.al.|[2410.14649](http://arxiv.org/abs/2410.14649)|**[link](https://github.com/ist-daslab/evopress)**|
|**2024-11-04**|**DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models**|Shangqian Gao et.al.|[2410.11988](http://arxiv.org/abs/2410.11988)|null|
|**2024-11-12**|**Self-Data Distillation for Recovering Quality in Pruned Large Language Models**|Vithursan Thangarasa et.al.|[2410.09982](http://arxiv.org/abs/2410.09982)|null|
|**2024-10-11**|**Unity is Power: Semi-Asynchronous Collaborative Training of Large-Scale Models with Structured Pruning in Resource-Limited Clients**|Yan Li et.al.|[2410.08457](http://arxiv.org/abs/2410.08457)|null|
|**2024-10-11**|**Chip-Tuning: Classify Before Language Models Say**|Fangwei Zhu et.al.|[2410.06541](http://arxiv.org/abs/2410.06541)|**[link](https://github.com/qq-mm/chiptuning)**|
|**2024-11-04**|**Large Language Model Compression with Neural Architecture Search**|Rhea Sanjay Sukthanker et.al.|[2410.06479](http://arxiv.org/abs/2410.06479)|null|
|**2024-09-29**|**Investigating the Effect of Network Pruning on Performance and Interpretability**|Jonathan von Rad et.al.|[2409.19727](http://arxiv.org/abs/2409.19727)|null|
|**2024-10-30**|**Search for Efficient Large Language Models**|Xuan Shen et.al.|[2409.17372](http://arxiv.org/abs/2409.17372)|**[link](https://github.com/shawnricecake/search-llm)**|
|**2024-09-22**|**SPAQ-DL-SLAM: Towards Optimizing Deep Learning-based SLAM for Resource-Constrained Embedded Platforms**|Niraj Pudasaini et.al.|[2409.14515](http://arxiv.org/abs/2409.14515)|null|
|**2024-09-20**|**CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information**|Yuxin Wang et.al.|[2409.13199](http://arxiv.org/abs/2409.13199)|**[link](https://github.com/wyxscir/cfsp)**|
|**2024-09-17**|**KVPruner: Structural Pruning for Faster and Memory-Efficient Large Language Models**|Bo Lv et.al.|[2409.11057](http://arxiv.org/abs/2409.11057)|null|
|**2024-09-11**|**HESSO: Towards Automatic Efficient and User Friendly Any Neural Network Training and Pruning**|Tianyi Chen et.al.|[2409.09085](http://arxiv.org/abs/2409.09085)|**[link](https://github.com/microsoft/only_train_once)**|
|**2024-09-12**|**Structured Pruning for Efficient Visual Place Recognition**|Oliver Grainge et.al.|[2409.07834](http://arxiv.org/abs/2409.07834)|null|
|**2024-09-10**|**STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning**|Jaeseong Lee et.al.|[2409.06211](http://arxiv.org/abs/2409.06211)|null|
|**2024-09-05**|**TropNNC: Structured Neural Network Compression Using Tropical Geometry**|Konstantinos Fotopoulos et.al.|[2409.03945](http://arxiv.org/abs/2409.03945)|null|
|**2024-09-02**|**Edge AI: Evaluation of Model Compression Techniques for Convolutional Neural Networks**|Samer Francy et.al.|[2409.02134](http://arxiv.org/abs/2409.02134)|null|
|**2024-08-27**|**PAT: Pruning-Aware Tuning for Large Language Models**|Yijiang Liu et.al.|[2408.14721](http://arxiv.org/abs/2408.14721)|**[link](https://github.com/kriskrisliu/pat_pruning-aware-tuning)**|
|**2024-08-15**|**PQV-Mobile: A Combined Pruning and Quantization Toolkit to Optimize Vision Transformers for Mobile Applications**|Kshitij Bhardwaj et.al.|[2408.08437](http://arxiv.org/abs/2408.08437)|**[link](https://github.com/kshitij11/pqv-mobile)**|
|**2024-08-13**|**Hybrid SD: Edge-Cloud Collaborative Inference for Stable Diffusion Models**|Chenqian Yan et.al.|[2408.06646](http://arxiv.org/abs/2408.06646)|null|
|**2024-08-06**|**Comb, Prune, Distill: Towards Unified Pruning for Vision Model Compression**|Jonas Schmitt et.al.|[2408.03046](http://arxiv.org/abs/2408.03046)|**[link](https://github.com/cranken/cpd)**|
|**2024-08-02**|**Sustainable Diffusion-based Incentive Mechanism for Generative AI-driven Digital Twins in Industrial Cyber-Physical Systems**|Jinbo Wen et.al.|[2408.01173](http://arxiv.org/abs/2408.01173)|null|
|**2024-08-22**|**Diff-Cleanse: Identifying and Mitigating Backdoor Attacks in Diffusion Models**|Jiang Hao et.al.|[2407.21316](http://arxiv.org/abs/2407.21316)|**[link](https://github.com/shymuel/diff-cleanse)**|
|**2024-07-26**|**Greedy Output Approximation: Towards Efficient Structured Pruning for LLMs Without Retraining**|Jianwei Li et.al.|[2407.19126](http://arxiv.org/abs/2407.19126)|null|
|**2024-07-17**|**MCU-MixQ: A HW/SW Co-optimized Mixed-precision Neural Network Design Framework for MCUs**|Junfeng Gong et.al.|[2407.18267](http://arxiv.org/abs/2407.18267)|null|
|**2024-07-24**|**(PASS) Visual Prompt Locates Good Structure Sparsity through a Recurrent HyperNetwork**|Tianjin Huang et.al.|[2407.17412](http://arxiv.org/abs/2407.17412)|null|
|**2024-07-22**|**Comprehensive Study on Performance Evaluation and Optimization of Model Compression: Bridging Traditional Deep Learning and Large Language Models**|Aayush Saxena et.al.|[2407.15904](http://arxiv.org/abs/2407.15904)|null|
|**2024-07-19**|**Shapley Pruning for Neural Network Compression**|Kamil Adamczewski et.al.|[2407.15875](http://arxiv.org/abs/2407.15875)|null|
|**2024-07-22**|**A Pairwise Comparison Relation-assisted Multi-objective Evolutionary Neural Architecture Search Method with Multi-population Mechanism**|Yu Xue et.al.|[2407.15600](http://arxiv.org/abs/2407.15600)|null|
|**2024-07-19**|**Straightforward Layer-wise Pruning for More Efficient Visual Adaptation**|Ruizi Han et.al.|[2407.14330](http://arxiv.org/abs/2407.14330)|null|
|**2024-07-18**|**Data-Algorithm-Architecture Co-Optimization for Fair Neural Networks on Skin Lesion Dataset**|Yi Sheng et.al.|[2407.13896](http://arxiv.org/abs/2407.13896)|null|
|**2024-07-18**|**Reconstruct the Pruned Model without Any Retraining**|Pingjie Wang et.al.|[2407.13331](http://arxiv.org/abs/2407.13331)|null|
|**2024-07-18**|**MO-EMT-NAS: Multi-Objective Continuous Transfer of Architectural Knowledge Between Tasks from Different Datasets**|Peng Liao et.al.|[2407.13122](http://arxiv.org/abs/2407.13122)|null|
|**2024-07-16**|**MINI-LLM: Memory-Efficient Structured Pruning for Large Language Models**|Hongrong Cheng et.al.|[2407.11681](http://arxiv.org/abs/2407.11681)|null|
|**2024-07-15**|**DDFAD: Dataset Distillation Framework for Audio Data**|Wenbo Jiang et.al.|[2407.10446](http://arxiv.org/abs/2407.10446)|null|

<p align=right>(<a href=#updated-on-20241231>back to top</a>)</p>

## Hardware-Software Co-Design

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-12-29**|**A Novel FPGA-based CNN Hardware Accelerator: Optimization for Convolutional Layers using Karatsuba Ofman Multiplier**|Amit Sarkar et.al.|[2412.20393](http://arxiv.org/abs/2412.20393)|null|
|**2024-12-29**|**Open-Source Heterogeneous SoCs for AI: The PULP Platform Experience**|Francesco Conti et.al.|[2412.20391](http://arxiv.org/abs/2412.20391)|null|
|**2024-12-27**|**HADES: Hardware Accelerated Decoding for Efficient Speculation in Large Language Models**|Ze Yang et.al.|[2412.19925](http://arxiv.org/abs/2412.19925)|null|
|**2024-12-26**|**Evolution, Challenges, and Optimization in Computer Architecture: The Role of Reconfigurable Systems**|Jefferson Ederhion et.al.|[2412.19234](http://arxiv.org/abs/2412.19234)|null|
|**2024-12-24**|**GCN-ABFT: Low-Cost Online Error Checking for Graph Convolutional Networks**|Christodoulos Peltekis et.al.|[2412.18534](http://arxiv.org/abs/2412.18534)|null|
|**2024-12-23**|**Advantages of density in tensor network geometries for gradient based training**|Sergi Masot-Llima et.al.|[2412.17497](http://arxiv.org/abs/2412.17497)|null|
|**2024-12-20**|**Chorba: A novel CRC32 implementation**|Sam Russell et.al.|[2412.16398](http://arxiv.org/abs/2412.16398)|null|
|**2024-12-20**|**Designing Visual Explanations and Learner Controls to Engage Adolescents in AI-Supported Exercise Selection**|Jeroen Ooge et.al.|[2412.16034](http://arxiv.org/abs/2412.16034)|null|
|**2024-12-20**|**A survey on FPGA-based accelerator for ML models**|Feng Yan et.al.|[2412.15666](http://arxiv.org/abs/2412.15666)|null|
|**2024-12-19**|**LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-simulation**|Chenxu Zhou et.al.|[2412.15199](http://arxiv.org/abs/2412.15199)|null|
|**2024-12-18**|**Pattern Matching in AI Compilers and its Formalization (Extended Version)**|Joseph W. Cutler et.al.|[2412.13398](http://arxiv.org/abs/2412.13398)|null|
|**2024-12-17**|**if-ZKP: Intel FPGA-Based Acceleration of Zero Knowledge Proofs**|Shahzad Ahmad Butt et.al.|[2412.12481](http://arxiv.org/abs/2412.12481)|null|
|**2024-12-13**|**Strong Structural Bounds for MaxSAT: The Fine Details of Using Neuromorphic and Quantum Hardware Accelerators**|Max Bannach et.al.|[2412.10289](http://arxiv.org/abs/2412.10289)|null|
|**2024-12-16**|**MVQ:Towards Efficient DNN Compression and Acceleration with Masked Vector Quantization**|Shuaiting Li et.al.|[2412.10261](http://arxiv.org/abs/2412.10261)|null|
|**2024-12-12**|**MPAX: Mathematical Programming in JAX**|Haihao Lu et.al.|[2412.09734](http://arxiv.org/abs/2412.09734)|**[link](https://github.com/mit-lu-lab/mpax)**|
|**2024-12-12**|**Evaluating the Potential of In-Memory Processing to Accelerate Homomorphic Encryption**|Mpoki Mwaisela et.al.|[2412.09144](http://arxiv.org/abs/2412.09144)|null|
|**2024-12-12**|**Analyzing Practical Policies for Multiresource Job Scheduling**|Zhongrui Chen et.al.|[2412.08915](http://arxiv.org/abs/2412.08915)|null|
|**2024-12-09**|**LLM-BIP: Structured Pruning for Large Language Models with Block-Wise Forward Importance Propagation**|Haihang Wu et.al.|[2412.06419](http://arxiv.org/abs/2412.06419)|null|
|**2024-12-03**|**Demonstrating the Advantages of Analog Wafer-Scale Neuromorphic Hardware**|Hartmut Schmidt et.al.|[2412.02619](http://arxiv.org/abs/2412.02619)|null|
|**2024-12-03**|**Multi-timescale synaptic plasticity on analog neuromorphic hardware**|Amani Atoui et.al.|[2412.02515](http://arxiv.org/abs/2412.02515)|null|
|**2024-11-27**|**Deterministic and Probabilistic Rounding Error Analysis for Mixed-Precision Arithmetic on Modern Computing Units**|Sahil Bhola et.al.|[2411.18747](http://arxiv.org/abs/2411.18747)|null|
|**2024-11-26**|**Scalable iterative pruning of large language and vision models using block coordinate descent**|Gili Rosenberg et.al.|[2411.17796](http://arxiv.org/abs/2411.17796)|null|
|**2024-11-25**|**Limitations of tensor network approaches for optimization and sampling: A comparison against quantum and classical Ising machines**|Anna Maria Dziubyna et.al.|[2411.16431](http://arxiv.org/abs/2411.16431)|null|
|**2024-11-25**|**MixPE: Quantization and Hardware Co-design for Efficient LLM Inference**|Yu Zhang et.al.|[2411.16158](http://arxiv.org/abs/2411.16158)|null|
|**2024-11-20**|**Hardware Accelerators for Artificial Intelligence**|S M Mojahidul Ahsan et.al.|[2411.13717](http://arxiv.org/abs/2411.13717)|null|
|**2024-11-20**|**Hardware Scaling Trends and Diminishing Returns in Large-Scale Distributed Training**|Jared Fernandez et.al.|[2411.13055](http://arxiv.org/abs/2411.13055)|null|
|**2024-11-19**|**FGP: Feature-Gradient-Prune for Efficient Convolutional Layer Pruning**|Qingsong Lv et.al.|[2411.12781](http://arxiv.org/abs/2411.12781)|**[link](https://github.com/fgp-code/fgp)**|
|**2024-11-19**|**Design of an FPGA-Based Neutral Atom Rearrangement Accelerator for Quantum Computing**|Xiaorang Guo et.al.|[2411.12401](http://arxiv.org/abs/2411.12401)|null|
|**2024-11-18**|**SILVIA: Automated Superword-Level Parallelism Exploitation via HLS-Specific LLVM Passes for Compute-Intensive FPGA Accelerators**|Giovanni Brignone et.al.|[2411.11384](http://arxiv.org/abs/2411.11384)|**[link](https://github.com/brigio345/SILVIA)**|
|**2024-12-01**|**InvestESG: A multi-agent reinforcement learning benchmark for studying climate investment as a social dilemma**|Xiaoxuan Hou et.al.|[2411.09856](http://arxiv.org/abs/2411.09856)|**[link](https://github.com/yuanjiayiy/InvestESG)**|
|**2024-11-21**|**OpenGeMM: A High-Utilization GeMM Accelerator Generator with Lightweight RISC-V Control and Tight Memory Coupling**|Xiaoling Yi et.al.|[2411.09543](http://arxiv.org/abs/2411.09543)|null|
|**2024-11-15**|**Communication Compression for Tensor Parallel LLM Inference**|Jan Hansen-Palmus et.al.|[2411.09510](http://arxiv.org/abs/2411.09510)|null|
|**2024-11-18**|**RPCAcc: A High-Performance and Reconfigurable PCIe-attached RPC Accelerator**|Jie Zhang et.al.|[2411.07632](http://arxiv.org/abs/2411.07632)|null|
|**2024-11-11**|**Spiking Transformer Hardware Accelerators in 3D Integration**|Boxun Xu et.al.|[2411.07397](http://arxiv.org/abs/2411.07397)|null|
|**2024-11-10**|**AMAZE: Accelerated MiMC Hardware Architecture for Zero-Knowledge Applications on the Edge**|Anees Ahmed et.al.|[2411.06350](http://arxiv.org/abs/2411.06350)|**[link](https://github.com/ACES-STAM/AMAZE)**|
|**2024-11-03**|**Stochastic Communication Avoidance for Recommendation Systems**|Lutfi Eren Erdogan et.al.|[2411.01611](http://arxiv.org/abs/2411.01611)|null|
|**2024-11-01**|**Inducing Semi-Structured Sparsity by Masking for Efficient Model Inference in Convolutional Networks**|David A. Danhofer et.al.|[2411.00288](http://arxiv.org/abs/2411.00288)|null|
|**2024-10-31**|**LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators**|Krishna Teja Chitty-Venkata et.al.|[2411.00136](http://arxiv.org/abs/2411.00136)|**[link](https://github.com/argonne-lcf/llm-inference-bench)**|
|**2024-10-30**|**Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks**|Michael Matthews et.al.|[2410.23208](http://arxiv.org/abs/2410.23208)|**[link](https://github.com/michaeltmatthews/jax2d)**|
|**2024-10-24**|**Watermarking Large Language Models and the Generated Content: Opportunities and Challenges**|Ruisi Zhang et.al.|[2410.19096](http://arxiv.org/abs/2410.19096)|null|
|**2024-10-21**|**Hacking the Fabric: Targeting Partial Reconfiguration for Fault Injection in FPGA Fabrics**|Jayeeta Chaudhuri et.al.|[2410.16497](http://arxiv.org/abs/2410.16497)|null|
|**2024-10-21**|**Hyperparameter Optimisation in Deep Learning from Ensemble Methods: Applications to Proton Structure**|Juan Cruz-Martinez et.al.|[2410.16248](http://arxiv.org/abs/2410.16248)|null|
|**2024-10-20**|**A Remedy to Compute-in-Memory with Dynamic Random Access Memory: 1FeFET-1C Technology for Neuro-Symbolic AI**|Xunzhao Yin et.al.|[2410.15296](http://arxiv.org/abs/2410.15296)|null|
|**2024-10-18**|**Self-Satisfied: An end-to-end framework for SAT generation and prediction**|Christopher R. Serrano et.al.|[2410.14888](http://arxiv.org/abs/2410.14888)|null|
|**2024-10-17**|**Quamba: A Post-Training Quantization Recipe for Selective State Space Models**|Hung-Yueh Chiang et.al.|[2410.13229](http://arxiv.org/abs/2410.13229)|**[link](https://github.com/enyac-group/quamba)**|
|**2024-10-16**|**Mixed-precision finite element kernels and assembly: Rounding error analysis and hardware acceleration**|M. Croci et.al.|[2410.12614](http://arxiv.org/abs/2410.12614)|**[link](https://github.com/croci/mpfem-paper-experiments-2024)**|
|**2024-10-15**|**Fast Local Neural Regression for Low-Cost, Path Traced Lambertian Global Illumination**|Arturo Salmi et.al.|[2410.11625](http://arxiv.org/abs/2410.11625)|null|
|**2024-10-15**|**Efficiera Residual Networks: Hardware-Friendly Fully Binary Weight with 2-bit Activation Model Achieves Practical ImageNet Accuracy**|Shuntaro Takahashi et.al.|[2410.11553](http://arxiv.org/abs/2410.11553)|**[link](https://github.com/leapmind/ern)**|
|**2024-10-14**|**Differentiable Weightless Neural Networks**|Alan T. L. Bacellar et.al.|[2410.11112](http://arxiv.org/abs/2410.11112)|**[link](https://github.com/alanbacellar/DWN)**|
|**2024-10-14**|**SLaNC: Static LayerNorm Calibration**|Mahsa Salmani et.al.|[2410.10553](http://arxiv.org/abs/2410.10553)|null|
|**2024-10-11**|**MATCH: Model-Aware TVM-based Compilation for Heterogeneous Edge Devices**|Mohamed Amine Hamdi et.al.|[2410.08855](http://arxiv.org/abs/2410.08855)|**[link](https://github.com/eml-eda/match)**|
|**2024-10-09**|**Optimized Spatial Architecture Mapping Flow for Transformer Accelerators**|Haocheng Xu et.al.|[2410.07407](http://arxiv.org/abs/2410.07407)|null|
|**2024-10-09**|**Unlocking Real-Time Fluorescence Lifetime Imaging: Multi-Pixel Parallelism for FPGA-Accelerated Processing**|Ismail Erbas et.al.|[2410.07364](http://arxiv.org/abs/2410.07364)|null|
|**2024-10-03**|**CAX: Cellular Automata Accelerated in JAX**|Maxence Faldor et.al.|[2410.02651](http://arxiv.org/abs/2410.02651)|**[link](https://github.com/maxencefaldor/cax)**|
|**2024-10-03**|**Extracting the Potential of Emerging Hardware Accelerators for Symmetric Eigenvalue Decomposition**|Hansheng Wang et.al.|[2410.02170](http://arxiv.org/abs/2410.02170)|null|
|**2024-10-01**|**Compressing Recurrent Neural Networks for FPGA-accelerated Implementation in Fluorescence Lifetime Imaging**|Ismail Erbas et.al.|[2410.00948](http://arxiv.org/abs/2410.00948)|null|
|**2024-09-26**|**Leader Selection and Follower Association for UE-centric Distributed Learning in Future Wireless Networks**|Saeedeh Parsaeefard et.al.|[2409.18268](http://arxiv.org/abs/2409.18268)|null|
|**2024-09-26**|**A 5T-2MTJ STT-assisted Spin Orbit Torque based Ternary Content Addressable Memory for Hardware Accelerators**|Siri Narla et.al.|[2409.17863](http://arxiv.org/abs/2409.17863)|null|
|**2024-09-24**|**Microsecond-Latency Feedback at a Particle Accelerator by Online Reinforcement Learning on Hardware**|Luca Scomparin et.al.|[2409.16177](http://arxiv.org/abs/2409.16177)|null|
|**2024-09-25**|**Ultra-low latency quantum-inspired machine learning predictors implemented on FPGA**|Lorenzo Borella et.al.|[2409.16075](http://arxiv.org/abs/2409.16075)|null|
|**2024-09-19**|**Enhancing Performance and Scalability of Large-Scale Recommendation Systems with Jagged Flash Attention**|Rengan Xu et.al.|[2409.15373](http://arxiv.org/abs/2409.15373)|null|
|**2024-09-23**|**Efficient Tabular Data Preprocessing of ML Pipelines**|Yu Zhu et.al.|[2409.14912](http://arxiv.org/abs/2409.14912)|null|
|**2024-09-21**|**FAMOUS: Flexible Accelerator for the Attention Mechanism of Transformer on UltraScale+ FPGAs**|Ehsan Kabir et.al.|[2409.14023](http://arxiv.org/abs/2409.14023)|null|
|**2024-09-21**|**ProTEA: Programmable Transformer Encoder Acceleration on FPGA**|Ehsan Kabir et.al.|[2409.13975](http://arxiv.org/abs/2409.13975)|null|
|**2024-09-23**|**Towards Efficient Neuro-Symbolic AI: From Workload Characterization to Hardware Architecture**|Zishen Wan et.al.|[2409.13153](http://arxiv.org/abs/2409.13153)|null|
|**2024-09-20**|**Learning to Compare Hardware Designs for High-Level Synthesis**|Yunsheng Bai et.al.|[2409.13138](http://arxiv.org/abs/2409.13138)|null|
|**2024-09-19**|**Performance and Power: Systematic Evaluation of AI Workloads on Accelerators with CARAML**|Chelsea Maria John et.al.|[2409.12994](http://arxiv.org/abs/2409.12994)|**[link](https://github.com/FZJ-JSC/CARAML)**|
|**2024-09-19**|**CrossRT: A cross platform programming technology for hardware-accelerated ray tracing in CG and CV applications**|Vladimir Frolov et.al.|[2409.12617](http://arxiv.org/abs/2409.12617)|null|
|**2024-09-15**|**Pack my weights and run! Minimizing overheads for in-memory computing accelerators**|Pouya Houshmand et.al.|[2409.11437](http://arxiv.org/abs/2409.11437)|null|
|**2024-09-11**|**Next-generation Probabilistic Computing Hardware with 3D MOSAICs, Illusion Scale-up, and Co-design**|Tathagata Srimani et.al.|[2409.11422](http://arxiv.org/abs/2409.11422)|null|
|**2024-09-09**|**Hardware Acceleration of Kolmogorov-Arnold Network (KAN) for Lightweight Edge Inference**|Wei-Hsing Huang et.al.|[2409.11418](http://arxiv.org/abs/2409.11418)|null|
|**2024-09-17**|**Dynamic Range Reduction via Branch-and-Bound**|Thore Gerlach et.al.|[2409.10863](http://arxiv.org/abs/2409.10863)|null|
|**2024-09-16**|**Count2Multiply: Reliable In-memory High-Radix Counting**|João Paulo Cardoso de Lima et.al.|[2409.10136](http://arxiv.org/abs/2409.10136)|null|
|**2024-09-16**|**Hardware-Accelerated Ray Tracing for Discrete and Continuous Collision Detection on GPUs**|Sizhe Sui et.al.|[2409.09918](http://arxiv.org/abs/2409.09918)|null|
|**2024-09-13**|**Distributed Binary Optimization with In-Memory Computing: An Application for the SAT Problem**|Xiangyi Zhang et.al.|[2409.09152](http://arxiv.org/abs/2409.09152)|null|
|**2024-09-13**|**Automatic Generation of Fast and Accurate Performance Models for Deep Neural Network Accelerators**|Konstantin Lübeck et.al.|[2409.08595](http://arxiv.org/abs/2409.08595)|null|
|**2024-09-17**|**Foragax: An Agent-Based Modelling Framework Based on JAX**|Siddharth Chaturvedi et.al.|[2409.06345](http://arxiv.org/abs/2409.06345)|**[link](https://github.com/i-m-iron-man/Foragax)**|
|**2024-09-10**|**PIM-MMU: A Memory Management Unit for Accelerating Data Transfers in Commercial PIM Systems**|Dongjae Lee et.al.|[2409.06204](http://arxiv.org/abs/2409.06204)|null|
|**2024-09-06**|**Towards Narrowing the Generalization Gap in Deep Boolean Networks**|Youngsung Kim et.al.|[2409.05905](http://arxiv.org/abs/2409.05905)|null|
|**2024-09-09**|**Supervised Learning for Stochastic Optimal Control**|Vince Kurtz et.al.|[2409.05792](http://arxiv.org/abs/2409.05792)|null|
|**2024-09-08**|**BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration**|Yuzong Chen et.al.|[2409.05227](http://arxiv.org/abs/2409.05227)|**[link](https://github.com/yc2367/bbs-micro)**|
|**2024-09-05**|**Libra: Architectural Support For Principled, Secure And Efficient Balanced Execution On High-End Processors (Extended Version)**|Hans Winderix et.al.|[2409.03743](http://arxiv.org/abs/2409.03743)|null|
|**2024-09-05**|**Hardware Acceleration of LLMs: A comprehensive survey and comparison**|Nikoletta Koilia et.al.|[2409.03384](http://arxiv.org/abs/2409.03384)|null|
|**2024-09-05**|**Towards training digitally-tied analog blocks via hybrid gradient computation**|Timothy Nest et.al.|[2409.03306](http://arxiv.org/abs/2409.03306)|null|
|**2024-08-30**|**The picasso gas model: Painting intracluster gas on gravity-only simulations**|F. Kéruzoré et.al.|[2408.17445](http://arxiv.org/abs/2408.17445)|**[link](https://github.com/fkeruzore/picasso)**|
|**2024-08-29**|**Serial and Parallel Two-Column Probing for Mixed-Integer Programming**|Yongzheng Dai et.al.|[2408.16927](http://arxiv.org/abs/2408.16927)|**[link](https://github.com/foreverdyz/twocolumnprobing)**|
|**2024-08-29**|**On-device AI: Quantization-aware Training of Transformers in Time-Series**|Tianheng Ling et.al.|[2408.16495](http://arxiv.org/abs/2408.16495)|null|
|**2024-08-29**|**Accelerating Image-based Pest Detection on a Heterogeneous Multi-core Microcontroller**|Luca Bompani et.al.|[2408.15911](http://arxiv.org/abs/2408.15911)|**[link](https://github.com/bomps4/tafe_pest_detection)**|
|**2024-08-28**|**FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration with Reconfigurable Spatial Architecture**|Tenglong Li et.al.|[2408.15578](http://arxiv.org/abs/2408.15578)|null|
|**2024-08-29**|**CGRA4ML: A Framework to Implement Modern Neural Networks for Scientific Edge Computing**|G Abarajithan et.al.|[2408.15561](http://arxiv.org/abs/2408.15561)|null|
|**2024-08-27**|**SCAN-Edge: Finding MobileNet-speed Hybrid Networks for Diverse Edge Devices via Hardware-Aware Evolutionary Search**|Hung-Yueh Chiang et.al.|[2408.15395](http://arxiv.org/abs/2408.15395)|null|
|**2024-08-27**|**SiHGNN: Leveraging Properties of Semantic Graphs for Efficient HGNN Acceleration**|Runzhen Xue et.al.|[2408.15089](http://arxiv.org/abs/2408.15089)|null|
|**2024-08-26**|**On-Chip Learning with Memristor-Based Neural Networks: Assessing Accuracy and Efficiency Under Device Variations, Conductance Errors, and Input Noise**|M. Reza Eslami et.al.|[2408.14680](http://arxiv.org/abs/2408.14680)|null|
|**2024-08-26**|**HAPM -- Hardware Aware Pruning Method for CNN hardware accelerators in resource constrained devices**|Federico Nicolas Peccia et.al.|[2408.14055](http://arxiv.org/abs/2408.14055)|null|
|**2024-08-22**|**Hardware Acceleration for Knowledge Graph Processing: Challenges & Recent Developments**|Maciej Besta et.al.|[2408.12173](http://arxiv.org/abs/2408.12173)|null|
|**2024-08-21**|**Floating-Point Multiply-Add with Approximate Normalization for Low-Cost Matrix Engines**|Kosmas Alexandridis et.al.|[2408.11997](http://arxiv.org/abs/2408.11997)|null|
|**2024-08-21**|**Cage: Hardware-Accelerated Safe WebAssembly**|Martin Fink et.al.|[2408.11456](http://arxiv.org/abs/2408.11456)|null|
|**2024-08-20**|**Tapping in a Remote Vehicle's onboard LLM to Complement the Ego Vehicle's Field-of-View**|Malsha Ashani Mahawatta Dona et.al.|[2408.10794](http://arxiv.org/abs/2408.10794)|null|
|**2024-08-16**|**Xpikeformer: Hybrid Analog-Digital Hardware Acceleration for Spiking Transformers**|Zihang Song et.al.|[2408.08794](http://arxiv.org/abs/2408.08794)|null|
|**2024-08-16**|**Cross-Chip Partial Reconfiguration for the Initialisation of Modular and Scalable Heterogeneous Systems**|Marvin Fuchs et.al.|[2408.08626](http://arxiv.org/abs/2408.08626)|null|
|**2024-08-13**|**HLSPilot: LLM-based High-Level Synthesis**|Chenwei Xiong et.al.|[2408.06810](http://arxiv.org/abs/2408.06810)|**[link](https://github.com/xcw-1010/hlspilot)**|
|**2024-08-12**|**Hardware Architecture Design of Model-Based Image Reconstruction Towards Palm-size Photoacoustic Tomography**|Yuwei Zheng et.al.|[2408.06049](http://arxiv.org/abs/2408.06049)|null|
|**2024-08-12**|**SZKP: A Scalable Accelerator Architecture for Zero-Knowledge Proofs**|Alhad Daftardar et.al.|[2408.05890](http://arxiv.org/abs/2408.05890)|null|
|**2024-08-10**|**LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale**|Jaehong Cho et.al.|[2408.05499](http://arxiv.org/abs/2408.05499)|**[link](https://github.com/casys-kaist/llmservingsim)**|
|**2024-08-08**|**Noise-augmented Chaotic Ising Machines for Combinatorial Optimization and Sampling**|Kyle Lee et.al.|[2408.04744](http://arxiv.org/abs/2408.04744)|null|
|**2024-08-07**|**Hardware-Assisted Virtualization of Neural Processing Units for Cloud Platforms**|Yuqi Xue et.al.|[2408.04104](http://arxiv.org/abs/2408.04104)|null|
|**2024-08-07**|**Real-time Event Recognition of Long-distance Distributed Vibration Sensing with Knowledge Distillation and Hardware Acceleration**|Zhongyao Luo et.al.|[2408.03647](http://arxiv.org/abs/2408.03647)|**[link](https://github.com/hust-iof/efficient-dvs)**|
|**2024-08-06**|**LLM-Aided Compilation for Tensor Accelerators**|Charles Hong et.al.|[2408.03408](http://arxiv.org/abs/2408.03408)|null|
|**2024-08-06**|**HeTraX: Energy Efficient 3D Heterogeneous Manycore Architecture for Transformer Acceleration**|Pratyush Dhingra et.al.|[2408.03397](http://arxiv.org/abs/2408.03397)|null|
|**2024-08-05**|**PENDRAM: Enabling High-Performance and Energy-Efficient Processing of Deep Neural Networks through a Generalized DRAM Data Mapping Policy**|Rachmad Vidya Wicaksana Putra et.al.|[2408.02412](http://arxiv.org/abs/2408.02412)|null|
|**2024-08-02**|**Digitized Phase Change Material Heterostack for Diffractive Optical Neural Network**|Ruiyang Chen et.al.|[2408.01404](http://arxiv.org/abs/2408.01404)|null|
|**2024-08-02**|**Search-in-Memory (SiM): Reliable, Versatile, and Efficient Data Matching in SSD's NAND Flash Memory Chip for Data Indexing Acceleration**|Yun-Chih Chen et.al.|[2408.00327](http://arxiv.org/abs/2408.00327)|null|
|**2024-08-07**|**Temporal Feature Matters: A Framework for Diffusion Model Quantization**|Yushi Huang et.al.|[2407.19547](http://arxiv.org/abs/2407.19547)|null|
|**2024-07-16**|**Latency optimized Deep Neural Networks (DNNs): An Artificial Intelligence approach at the Edge using Multiprocessor System on Chip (MPSoC)**|Seyed Nima Omidsajedi et.al.|[2407.18264](http://arxiv.org/abs/2407.18264)|null|
|**2024-07-22**|**KWT-Tiny: RISC-V Accelerated, Embedded Keyword Spotting Transformer**|Aness Al-Qawlaq et.al.|[2407.16026](http://arxiv.org/abs/2407.16026)|null|
|**2024-07-18**|**Integrated Hardware Architecture and Device Placement Search**|Irene Wang et.al.|[2407.13143](http://arxiv.org/abs/2407.13143)|**[link](https://github.com/msr-fiddle/phaze)**|
|**2024-07-17**|**ARTEMIS: A Mixed Analog-Stochastic In-DRAM Accelerator for Transformer Neural Networks**|Salma Afifi et.al.|[2407.12638](http://arxiv.org/abs/2407.12638)|null|
|**2024-07-17**|**StoX-Net: Stochastic Processing of Partial Sums for Efficient In-Memory Computing DNN Accelerators**|Ethan G Rogers et.al.|[2407.12378](http://arxiv.org/abs/2407.12378)|null|
|**2024-07-16**|**Co-Designing Binarized Transformer and Hardware Accelerator for Efficient End-to-End Edge Deployment**|Yuhao Ji et.al.|[2407.12070](http://arxiv.org/abs/2407.12070)|null|
|**2024-07-16**|**Ascend-CC: Confidential Computing on Heterogeneous NPU for Emerging Generative AI Workloads**|Aritra Dhar et.al.|[2407.11888](http://arxiv.org/abs/2407.11888)|null|
|**2024-07-15**|**Hierarchical search method for gravitational waves from stellar-mass binary black holes in noisy space-based detector data**|Yao Fu et.al.|[2407.10797](http://arxiv.org/abs/2407.10797)|null|
|**2024-07-14**|**Accelerator-as-a-Service in Public Clouds: An Intra-Host Traffic Management View for Performance Isolation in the Wild**|Jiechen Zhao et.al.|[2407.10098](http://arxiv.org/abs/2407.10098)|null|
|**2024-07-12**|**68-Channel Highly-Integrated Neural Signal Processing PSoC with On-Chip Feature Extraction, Compression, and Hardware Accelerators for Neuroprosthetics in 22nm FDSOI**|Liyuan Guo et.al.|[2407.09166](http://arxiv.org/abs/2407.09166)|null|
|**2024-07-12**|**Hybrid Temporal Computing for Lower Power Hardware Accelerators**|Maliha Tasnim et.al.|[2407.08975](http://arxiv.org/abs/2407.08975)|null|

<p align=right>(<a href=#updated-on-20241231>back to top</a>)</p>

## TinyML

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-12-25**|**Tempus Core: Area-Power Efficient Temporal-Unary Convolution Core for Low-Precision Edge DLAs**|Prabhu Vellaisamy et.al.|[2412.19002](http://arxiv.org/abs/2412.19002)|null|
|**2024-12-23**|**Edge-AI for Agriculture: Lightweight Vision Models for Disease Detection in Resource-Limited Settings**|Harsh Joshi et.al.|[2412.18635](http://arxiv.org/abs/2412.18635)|null|
|**2024-12-23**|**tuGEMM: Area-Power-Efficient Temporal Unary GEMM Architecture for Low-Precision Edge AI**|Harideep Nair et.al.|[2412.17966](http://arxiv.org/abs/2412.17966)|null|
|**2024-12-22**|**Fatigue Monitoring Using Wearables and AI: Trends, Challenges, and Future Opportunities**|Kourosh Kakhi et.al.|[2412.16847](http://arxiv.org/abs/2412.16847)|null|
|**2024-12-19**|**ElectraSight: Smart Glasses with Fully Onboard Non-Invasive Eye Tracking Using Hybrid Contact and Contactless EOG**|Nicolas Schärer et.al.|[2412.14848](http://arxiv.org/abs/2412.14848)|null|
|**2024-12-17**|**Design of an AI-Enhanced Digital Stethoscope: Advancing Cardiovascular Diagnostics Through Smart Auscultation**|Abraham G. Taye et.al.|[2412.14206](http://arxiv.org/abs/2412.14206)|null|
|**2024-12-16**|**Flex-PE: Flexible and SIMD Multi-Precision Processing Element for AI Workloads**|Mukul Lokhande et.al.|[2412.11702](http://arxiv.org/abs/2412.11702)|**[link](https://github.com/mukullokhande99/cordic-code)**|
|**2024-12-13**|**Edge AI-based Radio Frequency Fingerprinting for IoT Networks**|Ahmed Mohamed Hussain et.al.|[2412.10553](http://arxiv.org/abs/2412.10553)|null|
|**2024-12-13**|**EI-Drive: A Platform for Cooperative Perception with Realistic Communication Models**|Hanchu Zhou et.al.|[2412.09782](http://arxiv.org/abs/2412.09782)|null|
|**2024-12-12**|**Optimising TinyML with Quantization and Distillation of Transformer and Mamba Models for Indoor Localisation on Edge Devices**|Thanaphon Suwannaphong et.al.|[2412.09289](http://arxiv.org/abs/2412.09289)|null|
|**2024-12-10**|**Performance Evaluation of ROS2-DDS middleware implementations facilitating Cooperative Driving in Autonomous Vehicle**|Sumit Paul et.al.|[2412.07485](http://arxiv.org/abs/2412.07485)|null|
|**2024-12-07**|**Innovative Sentiment Analysis and Prediction of Stock Price Using FinBERT, GPT-4 and Logistic Regression: A Data-Driven Approach**|Olamilekan Shobayo et.al.|[2412.06837](http://arxiv.org/abs/2412.06837)|null|
|**2024-12-09**|**DEX: Data Channel Extension for Efficient CNN Inference on Tiny AI Accelerators**|Taesik Gong et.al.|[2412.06566](http://arxiv.org/abs/2412.06566)|**[link](https://github.com/nokia-bell-labs/data-channel-extension)**|
|**2024-12-09**|**Sequential Printed MLP Circuits for Super TinyML Multi-Sensory Applications**|Gurol Saglam et.al.|[2412.06542](http://arxiv.org/abs/2412.06542)|null|
|**2024-12-02**|**Optimizing LoRa for Edge Computing with TinyML Pipeline for Channel Hopping**|Marla Grunewald et.al.|[2412.01609](http://arxiv.org/abs/2412.01609)|null|
|**2024-12-01**|**Toward Real-Time Edge AI: Model-Agnostic Task-Oriented Communication with Visual Feature Alignment**|Songjie Xie et.al.|[2412.00862](http://arxiv.org/abs/2412.00862)|**[link](https://github.com/SongjieXie/feature-alignment-TOC)**|
|**2024-11-28**|**Co-Learning: Towards Semi-Supervised Object Detection with Road-side Cameras**|Jicheng Yuan et.al.|[2411.19143](http://arxiv.org/abs/2411.19143)|null|
|**2024-11-28**|**Towards an Implementation of the Knowledge-Based Control Plane for Intelligent Swarm Networks**|Xuanchi Guo et.al.|[2411.19068](http://arxiv.org/abs/2411.19068)|null|
|**2024-11-24**|**Space-ground Fluid AI for 6G Edge Intelligence**|Qian Chen et.al.|[2411.15845](http://arxiv.org/abs/2411.15845)|null|
|**2024-11-20**|**Federated Continual Learning for Edge-AI: A Comprehensive Survey**|Zi Wang et.al.|[2411.13740](http://arxiv.org/abs/2411.13740)|null|
|**2024-11-16**|**Enhanced FIWARE-Based Architecture for Cyberphysical Systems With Tiny Machine Learning and Machine Learning Operations: A Case Study on Urban Mobility Systems**|Javier Conde et.al.|[2411.13583](http://arxiv.org/abs/2411.13583)|null|
|**2024-11-19**|**Signformer is all you need: Towards Edge AI for Sign Language**|Eta Yang et.al.|[2411.12901](http://arxiv.org/abs/2411.12901)|**[link](https://github.com/EtaEnding/Signformer)**|
|**2024-11-16**|**DEBUG-HD: Debugging TinyML models on-device using Hyper-Dimensional computing**|Nikhil P Ghanathe et.al.|[2411.10692](http://arxiv.org/abs/2411.10692)|null|
|**2024-11-14**|**ABCI 3.0: Evolution of the leading AI infrastructure in Japan**|Ryousei Takano et.al.|[2411.09134](http://arxiv.org/abs/2411.09134)|null|
|**2024-11-13**|**A Cost-effective, Stand-alone, and Real-time TinyML-Based Gait Diagnosis Unit Aimed at Lower-limb Robotic Prostheses and Exoskeletons**|Zarin Anjum Madhiha et.al.|[2411.08474](http://arxiv.org/abs/2411.08474)|null|
|**2024-11-12**|**Towards Vision Mixture of Experts for Wildlife Monitoring on the Edge**|Emmanuel Azuh Mensah et.al.|[2411.07834](http://arxiv.org/abs/2411.07834)|null|
|**2024-11-16**|**Enhancing Predictive Maintenance in Mining Mobile Machinery through a TinyML-enabled Hierarchical Inference Network**|Raúl de la Fuente et.al.|[2411.07168](http://arxiv.org/abs/2411.07168)|null|
|**2024-11-11**|**A Primer on Word Embeddings: AI Techniques for Text Analysis in Social Work**|Brian E. Perron et.al.|[2411.07156](http://arxiv.org/abs/2411.07156)|null|
|**2024-11-11**|**TinyML Security: Exploring Vulnerabilities in Resource-Constrained Machine Learning Systems**|Jacob Huckelberry et.al.|[2411.07114](http://arxiv.org/abs/2411.07114)|null|
|**2024-11-10**|**Activation Map Compression through Tensor Decomposition for Deep Learning**|Le-Trung Nguyen et.al.|[2411.06346](http://arxiv.org/abs/2411.06346)|**[link](https://github.com/le-trungnguyen/neurips2024-activationcompression)**|
|**2024-11-09**|**TinyML NLP Approach for Semantic Wireless Sentiment Classification**|Ahmed Y. Radwan et.al.|[2411.06291](http://arxiv.org/abs/2411.06291)|null|
|**2024-11-03**|**Energy-Aware FPGA Implementation of Spiking Neural Network with LIF Neurons**|Asmer Hamid Ali et.al.|[2411.01628](http://arxiv.org/abs/2411.01628)|null|
|**2024-11-01**|**On the Impact of White-box Deployment Strategies for Edge AI on Latency and Model Performance**|Jaskirat Singh et.al.|[2411.00907](http://arxiv.org/abs/2411.00907)|null|
|**2024-10-30**|**Profiling AI Models: Towards Efficient Computation Offloading in Heterogeneous Edge AI Systems**|Juan Marcelo Parra-Ullauri et.al.|[2411.00859](http://arxiv.org/abs/2411.00859)|null|
|**2024-11-01**|**GPT for Games: An Updated Scoping Review (2020-2024)**|Daijin Yang et.al.|[2411.00308](http://arxiv.org/abs/2411.00308)|null|
|**2024-10-31**|**Cough-E: A multimodal, privacy-preserving cough detection algorithm for the edge**|Stefano Albini et.al.|[2410.24066](http://arxiv.org/abs/2410.24066)|**[link](https://github.com/esl-epfl/Cough-E)**|
|**2024-10-28**|**FusedInf: Efficient Swapping of DNN Models for On-Demand Serverless Inference Services on the Edge**|Sifat Ut Taki et.al.|[2410.21120](http://arxiv.org/abs/2410.21120)|**[link](https://github.com/sifattaj/fusedinf)**|
|**2024-10-28**|**Edge Perception: Intelligent Wireless Sensing at Network Edge**|Yuanhao Cui et.al.|[2410.21017](http://arxiv.org/abs/2410.21017)|null|
|**2024-10-25**|**Neuromorphic IoT Architecture for Efficient Water Management: A Smart Village Case Study**|Mugdim Bublin et.al.|[2410.19562](http://arxiv.org/abs/2410.19562)|null|
|**2024-10-17**|**SouLLMate: An Application Enhancing Diverse Mental Health Support with Adaptive LLMs, Prompt Engineering, and RAG Techniques**|Qiming Guo et.al.|[2410.16322](http://arxiv.org/abs/2410.16322)|null|
|**2024-10-21**|**P-YOLOv8: Efficient and Accurate Real-Time Detection of Distracted Driving**|Mohamed R. Elshamy et.al.|[2410.15602](http://arxiv.org/abs/2410.15602)|null|
|**2024-10-15**|**SHAKTI: A 2.5 Billion Parameter Small Language Model Optimized for Edge AI and Low-Resource Environments**|Syed Abdul Gaffar Shakhadri et.al.|[2410.11331](http://arxiv.org/abs/2410.11331)|null|
|**2024-10-14**|**ABBA-VSM: Time Series Classification using Symbolic Representation on the Edge**|Meerzhan Kanatbekova et.al.|[2410.10285](http://arxiv.org/abs/2410.10285)|null|
|**2024-10-12**|**Token Pruning using a Lightweight Background Aware Vision Transformer**|Sudhakar Sah et.al.|[2410.09324](http://arxiv.org/abs/2410.09324)|null|
|**2024-10-11**|**MATCH: Model-Aware TVM-based Compilation for Heterogeneous Edge Devices**|Mohamed Amine Hamdi et.al.|[2410.08855](http://arxiv.org/abs/2410.08855)|**[link](https://github.com/eml-eda/match)**|
|**2024-10-11**|**Edge AI Collaborative Learning: Bayesian Approaches to Uncertainty Estimation**|Gleb Radchenko et.al.|[2410.08651](http://arxiv.org/abs/2410.08651)|null|
|**2024-10-10**|**Neural Architecture Search of Hybrid Models for NPU-CIM Heterogeneous AR/VR Devices**|Yiwei Zhao et.al.|[2410.08326](http://arxiv.org/abs/2410.08326)|null|
|**2024-10-10**|**L-VITeX: Light-weight Visual Intuition for Terrain Exploration**|Antar Mazumder et.al.|[2410.07872](http://arxiv.org/abs/2410.07872)|null|
|**2024-10-10**|**Towards Robust IoT Defense: Comparative Statistics of Attack Detection in Resource-Constrained Scenarios**|Zainab Alwaisi et.al.|[2410.07810](http://arxiv.org/abs/2410.07810)|null|
|**2024-10-10**|**vCLIC: Towards Fast Interrupt Handling in Virtualized RISC-V Mixed-criticality Systems**|Enrico Zelioli et.al.|[2410.07798](http://arxiv.org/abs/2410.07798)|null|
|**2024-10-07**|**SoK: Towards Security and Safety of Edge AI**|Tatjana Wingarz et.al.|[2410.05349](http://arxiv.org/abs/2410.05349)|null|
|**2024-10-10**|**SONAR: A Synthetic AI-Audio Detection Framework and Benchmark**|Xiang Li et.al.|[2410.04324](http://arxiv.org/abs/2410.04324)|**[link](https://github.com/jessegator/sonar)**|
|**2024-09-28**|**MicroFlow: An Efficient Rust-Based Inference Engine for TinyML**|Matteo Carnelos et.al.|[2409.19432](http://arxiv.org/abs/2409.19432)|**[link](https://github.com/matteocarnelos/microflow-rs)**|
|**2024-09-27**|**Analog fast Fourier transforms for scalable and efficient signal processing**|T. Patrick Xiao et.al.|[2409.19071](http://arxiv.org/abs/2409.19071)|null|
|**2024-09-26**|**Development of an Edge Resilient ML Ensemble to Tolerate ICS Adversarial Attacks**|Likai Yao et.al.|[2409.18244](http://arxiv.org/abs/2409.18244)|null|
|**2024-09-25**|**Susceptibility Formulation of Density Matrix Perturbation Theory**|Anders M. N. Niklasson et.al.|[2409.17033](http://arxiv.org/abs/2409.17033)|null|
|**2024-09-25**|**Ethical and Scalable Automation: A Governance and Compliance Framework for Business Applications**|Haocheng Lin et.al.|[2409.16872](http://arxiv.org/abs/2409.16872)|null|
|**2024-09-25**|**Accelerating TinyML Inference on Microcontrollers through Approximate Kernels**|Giorgos Armeniakos et.al.|[2409.16815](http://arxiv.org/abs/2409.16815)|**[link](https://github.com/GeorgeMentzos/ATAMAN-AuTo-driven-Approximation-and-Microcontroller-AcceleratioN-Toolkit)**|
|**2024-09-23**|**Benchmarking Edge AI Platforms for High-Performance ML Inference**|Rakshith Jayanth et.al.|[2409.14803](http://arxiv.org/abs/2409.14803)|null|
|**2024-09-24**|**CamelEval: Advancing Culturally Aligned Arabic Language Models and Benchmarks**|Zhaozhi Qian et.al.|[2409.12623](http://arxiv.org/abs/2409.12623)|null|
|**2024-09-17**|**AI Suggestions Homogenize Writing Toward Western Styles and Diminish Cultural Nuances**|Dhruv Agarwal et.al.|[2409.11360](http://arxiv.org/abs/2409.11360)|null|
|**2024-09-17**|**Optimizing TinyML: The Impact of Reduced Data Acquisition Rates for Time Series Classification on Microcontrollers**|Riya Samanta et.al.|[2409.10942](http://arxiv.org/abs/2409.10942)|null|
|**2024-09-13**|**Pushing the boundaries of event subsampling in event-based video classification using CNNs**|Hesam Araghi et.al.|[2409.08953](http://arxiv.org/abs/2409.08953)|**[link](https://github.com/hesamaraghi/pushing-boundaries-event-subsampling)**|
|**2024-09-12**|**E-QUARTIC: Energy Efficient Edge Ensemble of Convolutional Neural Networks for Resource-Optimized Learning**|Le Zhang et.al.|[2409.08369](http://arxiv.org/abs/2409.08369)|null|
|**2024-09-12**|**DiReDi: Distillation and Reverse Distillation for AIoT Applications**|Chen Sun et.al.|[2409.08308](http://arxiv.org/abs/2409.08308)|null|
|**2024-09-11**|**A Continual and Incremental Learning Approach for TinyML On-device Training Using Dataset Distillation and Model Size Adaption**|Marcus Rüb et.al.|[2409.07114](http://arxiv.org/abs/2409.07114)|null|
|**2024-09-08**|**Transformer with Leveraged Masked Autoencoder for video-based Pain Assessment**|Minh-Duc Nguyen et.al.|[2409.05088](http://arxiv.org/abs/2409.05088)|null|
|**2024-09-02**|**Edge AI: Evaluation of Model Compression Techniques for Convolutional Neural Networks**|Samer Francy et.al.|[2409.02134](http://arxiv.org/abs/2409.02134)|null|
|**2024-09-01**|**Research on LLM Acceleration Using the High-Performance RISC-V Processor "Xiangshan" (Nanhu Version) Based on the Open-Source Matrix Instruction Set Extension (Vector Dot Product)**|Xu-Hao Chen et.al.|[2409.00661](http://arxiv.org/abs/2409.00661)|null|
|**2024-08-26**|**Towards Sustainable Personalized On-Device Human Activity Recognition with TinyML and Cloud-Enabled Auto Deployment**|Bidyut Saha et.al.|[2409.00093](http://arxiv.org/abs/2409.00093)|null|
|**2024-08-29**|**TinyTNAS: GPU-Free, Time-Bound, Hardware-Aware Neural Architecture Search for TinyML Time Series Classification**|Bidyut Saha et.al.|[2408.16535](http://arxiv.org/abs/2408.16535)|**[link](https://github.com/bidyutsaha/tinytnas)**|
|**2024-08-08**|**An Edge AI System Based on FPGA Platform for Railway Fault Detection**|Jiale Li et.al.|[2408.15245](http://arxiv.org/abs/2408.15245)|null|
|**2024-08-23**|**S3Simulator: A benchmarking Side Scan Sonar Simulator dataset for Underwater Image Analysis**|Kamal Basha S et.al.|[2408.12833](http://arxiv.org/abs/2408.12833)|**[link](https://github.com/bashakamal/s3simulator)**|
|**2024-08-20**|**Pluto and Charon: A Time and Memory Efficient Collaborative Edge AI Framework for Personal LLMs Fine-Tuning**|Bei Ouyang et.al.|[2408.10746](http://arxiv.org/abs/2408.10746)|null|
|**2024-08-21**|**Challenges and Responses in the Practice of Large Language Models**|Hongyin Zhu et.al.|[2408.09416](http://arxiv.org/abs/2408.09416)|null|
|**2024-08-15**|**Moving Healthcare AI-Support Systems for Visually Detectable Diseases onto Constrained Devices**|Tess Watt et.al.|[2408.08215](http://arxiv.org/abs/2408.08215)|null|
|**2024-08-14**|**Efficient Edge AI: Deploying Convolutional Neural Networks on FPGA with the Gemmini Accelerator**|Federico Nicolas Peccia et.al.|[2408.07404](http://arxiv.org/abs/2408.07404)|null|
|**2024-08-13**|**Harnessing Earnings Reports for Stock Predictions: A QLoRA-Enhanced LLM Approach**|Haowei Ni et.al.|[2408.06634](http://arxiv.org/abs/2408.06634)|null|
|**2024-08-06**|**Training on the Fly: On-device Self-supervised Learning aboard Nano-drones within 20 mW**|Elia Cereda et.al.|[2408.03168](http://arxiv.org/abs/2408.03168)|null|
|**2024-08-05**|**Toward Attention-based TinyML: A Heterogeneous Accelerated Architecture and Automated Deployment Flow**|Philip Wiese et.al.|[2408.02473](http://arxiv.org/abs/2408.02473)|null|
|**2024-08-05**|**PENDRAM: Enabling High-Performance and Energy-Efficient Processing of Deep Neural Networks through a Generalized DRAM Data Mapping Policy**|Rachmad Vidya Wicaksana Putra et.al.|[2408.02412](http://arxiv.org/abs/2408.02412)|null|
|**2024-08-02**|**A Tiny Supervised ODL Core with Auto Data Pruning for Human Activity Recognition**|Hiroki Matsutani et.al.|[2408.01283](http://arxiv.org/abs/2408.01283)|null|
|**2024-07-29**|**HOAA: Hybrid Overestimating Approximate Adder for Enhanced Performance Processing Engine**|Omkar Kokane et.al.|[2408.00806](http://arxiv.org/abs/2408.00806)|**[link](https://github.com/OmkarRajeshKokane/HOAA)**|
|**2024-07-31**|**TinyChirp: Bird Song Recognition Using TinyML Models on Low-power Wireless Acoustic Sensors**|Zhaolan Huang et.al.|[2407.21453](http://arxiv.org/abs/2407.21453)|**[link](https://github.com/TinyPART/TinyBirdSounds)**|
|**2024-07-31**|**SHA-CNN: Scalable Hierarchical Aware Convolutional Neural Network for Edge AI**|Narendra Singh Dhakad et.al.|[2407.21370](http://arxiv.org/abs/2407.21370)|null|
|**2024-07-30**|**On-the-fly Communication-and-Computing to Enable Representation Learning for Distributed Point Clouds**|Xu Chen et.al.|[2407.20710](http://arxiv.org/abs/2407.20710)|null|
|**2024-07-29**|**Model Agnostic Hybrid Sharding For Heterogeneous Distributed Inference**|Claudio Angione et.al.|[2407.19775](http://arxiv.org/abs/2407.19775)|null|
|**2024-07-25**|**A Sensitivity Analysis of Cellular Automata and Heterogeneous Topology Networks: Partially-Local Cellular Automata and Homogeneous Homogeneous Random Boolean Networks**|Tom Eivind Glover et.al.|[2407.18017](http://arxiv.org/abs/2407.18017)|null|
|**2024-07-22**|**StreamTinyNet: video streaming analysis with spatial-temporal TinyML**|Hazem Hesham Yousef Shalby et.al.|[2407.17524](http://arxiv.org/abs/2407.17524)|null|
|**2024-07-22**|**KWT-Tiny: RISC-V Accelerated, Embedded Keyword Spotting Transformer**|Aness Al-Qawlaq et.al.|[2407.16026](http://arxiv.org/abs/2407.16026)|null|
|**2024-07-18**|**Automated and Holistic Co-design of Neural Networks and ASICs for Enabling In-Pixel Intelligence**|Shubha R. Kharel et.al.|[2407.14560](http://arxiv.org/abs/2407.14560)|null|
|**2024-07-18**|**Ultra-Low-Latency Edge Inference for Distributed Sensing**|Zhanwei Wang et.al.|[2407.13360](http://arxiv.org/abs/2407.13360)|null|
|**2024-07-17**|**Computing: Looking Back and Moving Forward**|Muhammed Golec et.al.|[2407.12558](http://arxiv.org/abs/2407.12558)|null|
|**2024-07-16**|**XEdgeAI: A Human-centered Industrial Inspection Framework with Data-centric Explainable Edge AI Approach**|Truong Thanh Hung Nguyen et.al.|[2407.11771](http://arxiv.org/abs/2407.11771)|**[link](https://github.com/analytics-everywhere-lab/vqixai)**|
|**2024-07-18**|**Enhancing TinyML Security: Study of Adversarial Attack Transferability**|Parin Shah et.al.|[2407.11599](http://arxiv.org/abs/2407.11599)|null|
|**2024-07-13**|**Characterizing Disparity Between Edge Models and High-Accuracy Base Models for Vision Tasks**|Zhenyu Wang et.al.|[2407.10016](http://arxiv.org/abs/2407.10016)|null|
|**2024-07-11**|**Towards Efficient Deployment of Hybrid SNNs on Neuromorphic and Edge AI Hardware**|James Seekings et.al.|[2407.08704](http://arxiv.org/abs/2407.08704)|null|

<p align=right>(<a href=#updated-on-20241231>back to top</a>)</p>

## Domain Specific Accelerator

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-12-21**|**Leveraging Highly Approximated Multipliers in DNN Inference**|Georgios Zervakis et.al.|[2412.16757](http://arxiv.org/abs/2412.16757)|null|
|**2024-12-13**|**Panacea: Novel DNN Accelerator using Accuracy-Preserving Asymmetric Quantization and Energy-Saving Bit-Slice Sparsity**|Dongyun Kam et.al.|[2412.10059](http://arxiv.org/abs/2412.10059)|null|
|**2024-12-06**|**HiVeGen -- Hierarchical LLM-based Verilog Generation for Scalable Chip Design**|Jinwei Tang et.al.|[2412.05393](http://arxiv.org/abs/2412.05393)|null|
|**2024-12-06**|**MC3: Memory Contention based Covert Channel Communication on Shared DRAM System-on-Chips**|Ismet Dagli et.al.|[2412.05228](http://arxiv.org/abs/2412.05228)|null|
|**2024-11-28**|**PREBA: A Hardware/Software Co-Design for Multi-Instance GPU based AI Inference Servers**|Gwangoo Yeo et.al.|[2411.19114](http://arxiv.org/abs/2411.19114)|null|
|**2024-12-06**|**FAMES: Fast Approximate Multiplier Substitution for Mixed-Precision Quantized DNNs--Down to 2 Bits!**|Yi Ren et.al.|[2411.18055](http://arxiv.org/abs/2411.18055)|null|
|**2024-11-19**|**Travel Time Based Task Mapping for NoC-Based DNN Accelerator**|Yizhi Chen et.al.|[2411.12710](http://arxiv.org/abs/2411.12710)|null|
|**2024-10-29**|**Systolic Array Data Flows for Efficient Matrix Multiplication in Deep Neural Networks**|Tejas Raja et.al.|[2410.22595](http://arxiv.org/abs/2410.22595)|null|
|**2024-10-21**|**Adventures with Grace Hopper AI Super Chip and the National Research Platform**|J. Alex Hurt et.al.|[2410.16487](http://arxiv.org/abs/2410.16487)|null|
|**2024-10-17**|**Shavette: Low Power Neural Network Acceleration via Algorithm-level Error Detection and Undervolting**|Mikael Rinkinen et.al.|[2410.13415](http://arxiv.org/abs/2410.13415)|null|
|**2024-10-11**|**MATCH: Model-Aware TVM-based Compilation for Heterogeneous Edge Devices**|Mohamed Amine Hamdi et.al.|[2410.08855](http://arxiv.org/abs/2410.08855)|**[link](https://github.com/eml-eda/match)**|
|**2024-09-23**|**MESC: Re-thinking Algorithmic Priority and/or Criticality Inversions for Heterogeneous MCSs**|Jiapeng Guan et.al.|[2409.14837](http://arxiv.org/abs/2409.14837)|null|
|**2024-10-14**|**LoopTree: Exploring the Fused-layer Dataflow Accelerator Design Space**|Michael Gilbert et.al.|[2409.13625](http://arxiv.org/abs/2409.13625)|**[link](https://github.com/accelergy-project/looptree-tutorial)**|
|**2024-09-13**|**Automatic Generation of Fast and Accurate Performance Models for Deep Neural Network Accelerators**|Konstantin Lübeck et.al.|[2409.08595](http://arxiv.org/abs/2409.08595)|null|
|**2024-09-08**|**BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration**|Yuzong Chen et.al.|[2409.05227](http://arxiv.org/abs/2409.05227)|**[link](https://github.com/yc2367/bbs-micro)**|
|**2024-09-08**|**HYDRA: Hybrid Data Multiplexing and Run-time Layer Configurable DNN Accelerator**|Sonu Kumar et.al.|[2409.04976](http://arxiv.org/abs/2409.04976)|null|
|**2024-08-27**|**SiHGNN: Leveraging Properties of Semantic Graphs for Efficient HGNN Acceleration**|Runzhen Xue et.al.|[2408.15089](http://arxiv.org/abs/2408.15089)|null|
|**2024-08-24**|**SiTe CiM: Signed Ternary Computing-in-Memory for Ultra-Low Precision Deep Neural Networks**|Niharika Thakuria et.al.|[2408.13617](http://arxiv.org/abs/2408.13617)|null|
|**2024-08-13**|**Potamoi: Accelerating Neural Rendering via a Unified Streaming Architecture**|Yu Feng et.al.|[2408.06608](http://arxiv.org/abs/2408.06608)|null|
|**2024-09-24**|**Scaling Deep Learning Computation over the Inter-Core Connected Intelligence Processor with T10**|Yiqi Liu et.al.|[2408.04808](http://arxiv.org/abs/2408.04808)|null|
|**2024-07-30**|**Optical Computing for Deep Neural Network Acceleration: Foundations, Recent Developments, and Emerging Directions**|Sudeep Pasricha et.al.|[2407.21184](http://arxiv.org/abs/2407.21184)|null|
|**2024-07-29**|**Realizing Unaligned Block-wise Pruning for DNN Acceleration on Mobile Devices**|Hayun Lee et.al.|[2407.19644](http://arxiv.org/abs/2407.19644)|null|
|**2024-07-24**|**The Magnificent Seven Challenges and Opportunities in Domain-Specific Accelerator Design for Autonomous Systems**|Sabrina M. Neuman et.al.|[2407.17311](http://arxiv.org/abs/2407.17311)|null|
|**2024-07-17**|**StoX-Net: Stochastic Processing of Partial Sums for Efficient In-Memory Computing DNN Accelerators**|Ethan G Rogers et.al.|[2407.12378](http://arxiv.org/abs/2407.12378)|null|
|**2024-07-11**|**NinjaLLM: Fast, Scalable and Cost-effective RAG using Amazon SageMaker and AWS Trainium and Inferentia2**|Tengfei Xue et.al.|[2407.12057](http://arxiv.org/abs/2407.12057)|null|
|**2024-07-22**|**ARCO:Adaptive Multi-Agent Reinforcement Learning-Based Hardware/Software Co-Optimization Compiler for Improved Performance in DNN Accelerator Design**|Arya Fayyazi et.al.|[2407.08192](http://arxiv.org/abs/2407.08192)|null|
|**2024-06-20**|**SWANN: Shuffling Weights in Crossbar Arrays for Enhanced DNN Accuracy in Deeply Scaled Technologies**|Jeffry Victor et.al.|[2406.14706](http://arxiv.org/abs/2406.14706)|null|
|**2024-06-14**|**CMDS: Cross-layer Dataflow Optimization for DNN Accelerators Exploiting Multi-bank Memories**|Man Shi et.al.|[2406.14574](http://arxiv.org/abs/2406.14574)|null|
|**2024-06-15**|**Memory Faults in Activation-sparse Quantized Deep Neural Networks: Analysis and Mitigation using Sharpness-aware Training**|Akul Malhotra et.al.|[2406.10528](http://arxiv.org/abs/2406.10528)|null|
|**2024-07-17**|**Cross-Modality Program Representation Learning for Electronic Design Automation with High-Level Synthesis**|Zongyue Qin et.al.|[2406.09606](http://arxiv.org/abs/2406.09606)|null|
|**2024-06-05**|**HASS: Hardware-Aware Sparsity Search for Dataflow DNN Accelerator**|Zhewen Yu et.al.|[2406.03088](http://arxiv.org/abs/2406.03088)|**[link](https://github.com/yu-zhewen/hass)**|
|**2024-06-03**|**A 0.96pJ/SOP, 30.23K-neuron/mm^2 Heterogeneous Neuromorphic Chip With Fullerene-like Interconnection Topology for Edge-AI Computing**|P. J. Zhou et.al.|[2406.01151](http://arxiv.org/abs/2406.01151)|null|

<p align=right>(<a href=#updated-on-20241231>back to top</a>)</p>

## Low-Rank Adaptation

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-12-30**|**Adversarial Attack and Defense for LoRa Device Identification and Authentication via Deep Learning**|Yalin E. Sagduyu et.al.|[2412.21164](http://arxiv.org/abs/2412.21164)|null|
|**2024-12-30**|**Efficient Multi-Task Inferencing with a Shared Backbone and Lightweight Task-Specific Adapters for Automatic Scoring**|Ehsan Latif et.al.|[2412.21065](http://arxiv.org/abs/2412.21065)|null|
|**2024-12-30**|**DoTA: Weight-Decomposed Tensor Adaptation for Large Language Models**|Xiaolin Hu et.al.|[2412.20891](http://arxiv.org/abs/2412.20891)|null|
|**2024-12-30**|**Dual-Space Augmented Intrinsic-LoRA for Wind Turbine Segmentation**|Shubh Singhal et.al.|[2412.20838](http://arxiv.org/abs/2412.20838)|null|
|**2024-12-30**|**VMix: Improving Text-to-Image Diffusion Model with Cross-Attention Mixing Control**|Shaojin Wu et.al.|[2412.20800](http://arxiv.org/abs/2412.20800)|**[link](https://github.com/fenfenfenfan/VMix)**|
|**2024-12-29**|**EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers**|Daiheng Gao et.al.|[2412.20413](http://arxiv.org/abs/2412.20413)|null|
|**2024-12-28**|**Multi-Modality Driven LoRA for Adverse Condition Depth Estimation**|Guanglei Yang et.al.|[2412.20162](http://arxiv.org/abs/2412.20162)|null|
|**2024-12-28**|**VELoRA: A Low-Rank Adaptation Approach for Efficient RGB-Event based Recognition**|Lan Chen et.al.|[2412.20064](http://arxiv.org/abs/2412.20064)|**[link](https://github.com/event-ahu/velora)**|
|**2024-12-28**|**Adaptive Parameter-Efficient Federated Fine-Tuning on Heterogeneous Devices**|Jun Liu et.al.|[2412.20004](http://arxiv.org/abs/2412.20004)|null|
|**2024-12-27**|**Gradient Weight-normalized Low-rank Projection for Efficient LLM Training**|Jia-Hong Huang et.al.|[2412.19616](http://arxiv.org/abs/2412.19616)|null|
|**2024-12-27**|**Performance Evaluation of IoT LoRa Networks on Mars Through ns-3 Simulations**|Manuele Favero et.al.|[2412.19549](http://arxiv.org/abs/2412.19549)|**[link](https://github.com/manuelefavero/MarsPropagationLoss)**|
|**2024-12-27**|**KALAHash: Knowledge-Anchored Low-Resource Adaptation for Deep Hashing**|Shu Zhao et.al.|[2412.19417](http://arxiv.org/abs/2412.19417)|null|
|**2024-12-25**|**Optimizing Large Language Models with an Enhanced LoRA Fine-Tuning Algorithm for Efficiency and Robustness in NLP Tasks**|Jiacheng Hu et.al.|[2412.18729](http://arxiv.org/abs/2412.18729)|null|
|**2024-12-24**|**Research on the Proximity Relationships of Psychosomatic Disease Knowledge Graph Modules Extracted by Large Language Models**|Zihan Zhou et.al.|[2412.18419](http://arxiv.org/abs/2412.18419)|null|
|**2024-12-18**|**Enhancing Knowledge Distillation for LLMs with Response-Priming Prompting**|Vijay Goyal et.al.|[2412.17846](http://arxiv.org/abs/2412.17846)|**[link](https://github.com/alonso130r/knowledge-distillation)**|
|**2024-12-25**|**DreamFit: Garment-Centric Human Generation via a Lightweight Anything-Dressing Encoder**|Ente Lin et.al.|[2412.17644](http://arxiv.org/abs/2412.17644)|null|
|**2024-12-23**|**Resource-Aware Arabic LLM Creation: Model Adaptation, Integration, and Multi-Domain Testing**|Prakash Aryan et.al.|[2412.17548](http://arxiv.org/abs/2412.17548)|**[link](https://github.com/prakash-aryan/qwen-arabic-project)**|
|**2024-12-21**|**Label Privacy in Split Learning for Large Models with Parameter-Efficient Training**|Philip Zmushko et.al.|[2412.16669](http://arxiv.org/abs/2412.16669)|**[link](https://github.com/p3eft-iclr-2025/p3eft-iclr-2025)**|
|**2024-12-20**|**Adaptable and Precise: Enterprise-Scenario LLM Function-Calling Capability Training Pipeline**|Guancheng Zeng et.al.|[2412.15660](http://arxiv.org/abs/2412.15660)|null|
|**2024-12-23**|**CustomTTT: Motion and Appearance Customized Video Generation via Test-Time Training**|Xiuli Bi et.al.|[2412.15646](http://arxiv.org/abs/2412.15646)|**[link](https://github.com/rongpiking/customttt)**|
|**2024-12-20**|**AutoRank: MCDA Based Rank Personalization for LoRA-Enabled Distributed Learning**|Shuaijun Chen et.al.|[2412.15553](http://arxiv.org/abs/2412.15553)|null|
|**2024-12-19**|**Knowledge Injection via Prompt Distillation**|Kalle Kujanpää et.al.|[2412.14964](http://arxiv.org/abs/2412.14964)|null|
|**2024-12-20**|**All-in-One Tuning and Structural Pruning for Domain-Specific LLMs**|Lei Lu et.al.|[2412.14426](http://arxiv.org/abs/2412.14426)|null|
|**2024-12-18**|**CoRa: A Collision-Resistant LoRa Symbol Detector of Low Complexity**|José Álamos et.al.|[2412.13930](http://arxiv.org/abs/2412.13930)|null|
|**2024-12-18**|**A Comprehensive Evaluation of Parameter-Efficient Fine-Tuning on Method-Level Code Smell Detection**|Beiqi Zhang et.al.|[2412.13801](http://arxiv.org/abs/2412.13801)|**[link](https://github.com/mabelqi/peft4csd)**|
|**2024-12-18**|**Large Language Model Federated Learning with Blockchain and Unlearning for Cross-Organizational Collaboration**|Xuhan Zuo et.al.|[2412.13551](http://arxiv.org/abs/2412.13551)|null|
|**2024-12-18**|**Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models**|Xinxin Liu et.al.|[2412.13488](http://arxiv.org/abs/2412.13488)|null|
|**2024-12-18**|**Transducer Tuning: Efficient Model Adaptation for Software Tasks Using Code Property Graphs**|Imam Nur Bani Yusuf et.al.|[2412.13467](http://arxiv.org/abs/2412.13467)|**[link](https://github.com/imamnurby/transducer-tuning)**|
|**2024-12-17**|**Expansion Span: Combining Fading Memory and Retrieval in Hybrid State Space Models**|Elvis Nunez et.al.|[2412.13328](http://arxiv.org/abs/2412.13328)|null|
|**2024-12-17**|**FineGates: LLMs Finetuning with Compression using Stochastic Gates**|Jonathan Svirsky et.al.|[2412.12951](http://arxiv.org/abs/2412.12951)|null|
|**2024-12-17**|**Enhancing Naturalness in LLM-Generated Utterances through Disfluency Insertion**|Syed Zohaib Hassan et.al.|[2412.12710](http://arxiv.org/abs/2412.12710)|null|
|**2024-12-17**|**Train More Parameters But Mind Their Placement: Insights into Language Adaptation with PEFT**|Jenny Kunz et.al.|[2412.12674](http://arxiv.org/abs/2412.12674)|**[link](https://github.com/jekunz/peft-la)**|
|**2024-12-17**|**NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning**|Xin Yi et.al.|[2412.12497](http://arxiv.org/abs/2412.12497)|**[link](https://github.com/xinykou/nlsr)**|
|**2024-12-16**|**Visual Instruction Tuning with 500x Fewer Parameters through Modality Linear Representation-Steering**|Jinhe Bi et.al.|[2412.12359](http://arxiv.org/abs/2412.12359)|**[link](https://github.com/bibisbar/LLaVA-Steering)**|
|**2024-12-16**|**Can video generation replace cinematographers? Research on the cinematic language of generated video**|Xiaozhe Li et.al.|[2412.12223](http://arxiv.org/abs/2412.12223)|null|
|**2024-12-16**|**A LoRA is Worth a Thousand Pictures**|Chenxi Liu et.al.|[2412.12048](http://arxiv.org/abs/2412.12048)|null|
|**2024-12-16**|**The Open Source Advantage in Large Language Models (LLMs)**|Jiya Manchanda et.al.|[2412.12004](http://arxiv.org/abs/2412.12004)|null|
|**2024-12-17**|**No More Adam: Learning Rate Scaling at Initialization is All You Need**|Minghao Xu et.al.|[2412.11768](http://arxiv.org/abs/2412.11768)|**[link](https://github.com/anonymousalethiometer/sgd_sai)**|
|**2024-12-16**|**IDEA-Bench: How Far are Generative Models from Professional Designing?**|Chen Liang et.al.|[2412.11767](http://arxiv.org/abs/2412.11767)|**[link](https://github.com/ali-vilab/idea-bench)**|
|**2024-12-16**|**Adapting Segment Anything Model (SAM) to Experimental Datasets via Fine-Tuning on GAN-based Simulation: A Case Study in Additive Manufacturing**|Anika Tabassum et.al.|[2412.11381](http://arxiv.org/abs/2412.11381)|**[link](https://github.com/anikat1/sam-material-gan)**|
|**2024-12-16**|**FinLoRA: Finetuning Quantized Financial Large Language Models Using Low-Rank Adaptation**|Dannong Wang et.al.|[2412.11378](http://arxiv.org/abs/2412.11378)|null|
|**2024-12-15**|**Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models**|Di Wu et.al.|[2412.11041](http://arxiv.org/abs/2412.11041)|null|
|**2024-12-15**|**SceneLLM: Implicit Language Reasoning in LLM for Dynamic Scene Graph Generation**|Hang Zhang et.al.|[2412.11026](http://arxiv.org/abs/2412.11026)|null|
|**2024-12-14**|**Efficient Adaptation of Multilingual Models for Japanese ASR**|Mark Bajo et.al.|[2412.10705](http://arxiv.org/abs/2412.10705)|**[link](https://github.com/ryujimorita/tokyo_whisperers)**|
|**2024-12-13**|**SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation**|Runtao Liu et.al.|[2412.10493](http://arxiv.org/abs/2412.10493)|null|
|**2024-12-13**|**OP-LoRA: The Blessing of Dimensionality**|Piotr Teterwak et.al.|[2412.10362](http://arxiv.org/abs/2412.10362)|null|
|**2024-12-16**|**ASLoRA: Adaptive Sharing Low-Rank Adaptation Across Layers**|Junyan Hu et.al.|[2412.10135](http://arxiv.org/abs/2412.10135)|null|
|**2024-12-13**|**CaLoRAify: Calorie Estimation with Visual-Text Pairing and LoRA-Driven Visual Language Models**|Dongyu Yao et.al.|[2412.09936](http://arxiv.org/abs/2412.09936)|**[link](https://github.com/kennyyao2001/16824-caloraify)**|
|**2024-12-13**|**Low-Rank Adaptation with Task-Relevant Feature Enhancement for Fine-tuning Language Models**|Changqun Li et.al.|[2412.09827](http://arxiv.org/abs/2412.09827)|null|
|**2024-12-12**|**LoRACLR: Contrastive Adaptation for Customization of Diffusion Models**|Enis Simsar et.al.|[2412.09622](http://arxiv.org/abs/2412.09622)|null|
|**2024-12-12**|**EasyRef: Omni-Generalized Group Image Reference for Diffusion Models via Multimodal LLM**|Zhuofan Zong et.al.|[2412.09618](http://arxiv.org/abs/2412.09618)|null|
|**2024-12-12**|**Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition**|Zhisheng Zhong et.al.|[2412.09501](http://arxiv.org/abs/2412.09501)|**[link](https://github.com/dvlab-research/Lyra)**|
|**2024-12-15**|**GeLoRA: Geometric Adaptive Ranks For Efficient LoRA Fine-tuning**|Abdessalam Ed-dib et.al.|[2412.09250](http://arxiv.org/abs/2412.09250)|null|
|**2024-12-12**|**RAD: Region-Aware Diffusion Models for Image Inpainting**|Sora Kim et.al.|[2412.09191](http://arxiv.org/abs/2412.09191)|null|
|**2024-12-12**|**DECOR:Decomposition and Projection of Text Embeddings for Text-to-Image Customization**|Geonhui Jang et.al.|[2412.09169](http://arxiv.org/abs/2412.09169)|null|
|**2024-12-12**|**MoSLD: An Extremely Parameter-Efficient Mixture-of-Shared LoRAs for Multi-Task Learning**|Lulu Zhao et.al.|[2412.08946](http://arxiv.org/abs/2412.08946)|null|
|**2024-12-11**|**DMin: Scalable Training Data Influence Estimation for Diffusion Models**|Huawei Lin et.al.|[2412.08637](http://arxiv.org/abs/2412.08637)|**[link](https://github.com/huawei-lin/DMin)**|
|**2024-12-10**|**Accretion onto WD 2226 $-$ 210, the central star of the Helix Nebula**|S. Estrada-Dorado et.al.|[2412.07863](http://arxiv.org/abs/2412.07863)|null|
|**2024-12-10**|**PETALface: Parameter Efficient Transfer Learning for Low-resolution Face Recognition**|Kartik Narayan et.al.|[2412.07771](http://arxiv.org/abs/2412.07771)|null|
|**2024-12-10**|**LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation Models**|Ziqi Lu et.al.|[2412.07746](http://arxiv.org/abs/2412.07746)|null|
|**2024-12-10**|**ChocoLlama: Lessons Learned From Teaching Llamas Dutch**|Matthieu Meeus et.al.|[2412.07633](http://arxiv.org/abs/2412.07633)|null|
|**2024-12-10**|**MoDULA: Mixture of Domain-Specific and Universal LoRA for Multi-Task Learning**|Yufei Ma et.al.|[2412.07405](http://arxiv.org/abs/2412.07405)|null|
|**2024-12-10**|**Attention Head Purification: A New Perspective to Harness CLIP for Domain Generalization**|Yingfan Wang et.al.|[2412.07226](http://arxiv.org/abs/2412.07226)|null|
|**2024-12-09**|**Optimal Routing and Link Configuration for Covert Heterogeneous Wireless Networks**|Amna Gillani et.al.|[2412.07059](http://arxiv.org/abs/2412.07059)|null|
|**2024-12-09**|**Sequential Compression Layers for Efficient Federated Learning in Foundational Models**|Navyansh Mahla et.al.|[2412.07021](http://arxiv.org/abs/2412.07021)|null|
|**2024-12-09**|**BoRA: Bi-dimensional Weight-Decomposed Low-Rank Adaptation**|Qiushi Wang et.al.|[2412.06441](http://arxiv.org/abs/2412.06441)|null|
|**2024-12-10**|**S $^{2}$ FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity**|Xinyu Yang et.al.|[2412.06289](http://arxiv.org/abs/2412.06289)|null|
|**2024-12-08**|**Enhanced Computationally Efficient Long LoRA Inspired Perceiver Architectures for Auto-Regressive Language Modeling**|Kaleel Mahmood et.al.|[2412.06106](http://arxiv.org/abs/2412.06106)|null|
|**2024-12-08**|**KaSA: Knowledge-Aware Singular-Value Adaptation of Large Language Models**|Fan Wang et.al.|[2412.06071](http://arxiv.org/abs/2412.06071)|**[link](https://github.com/juyongjiang/kasa)**|
|**2024-12-07**|**Training-Free Bayesianization for Low-Rank Adapters of Large Language Models**|Haizhou Shi et.al.|[2412.05723](http://arxiv.org/abs/2412.05723)|**[link](https://github.com/wang-ml-lab/bayesian-peft)**|
|**2024-12-07**|**Plasmonic Electro-Optic Modulators based on Epsilon-Near-Zero Materials: Comparing the Classical Drift-Diffusion and Schrödinger-Poisson Coupling Models**|Masoud Shabaninezhad et.al.|[2412.05690](http://arxiv.org/abs/2412.05690)|null|
|**2024-12-06**|**QueEn: A Large Language Model for Quechua-English Translation**|Junhao Chen et.al.|[2412.05184](http://arxiv.org/abs/2412.05184)|null|
|**2024-12-06**|**LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation**|Donald Shenaj et.al.|[2412.05148](http://arxiv.org/abs/2412.05148)|null|
|**2024-12-05**|**Performance Evaluation of LoRa Technology for Rural Connectivity: An Experimental Analysis in Nepal**|Atit Pokharel et.al.|[2412.04563](http://arxiv.org/abs/2412.04563)|null|
|**2024-12-04**|**Prompting Large Language Models for Clinical Temporal Relation Extraction**|Jianping He et.al.|[2412.04512](http://arxiv.org/abs/2412.04512)|null|
|**2024-12-05**|**UnZipLoRA: Separating Content and Style from a Single Image**|Chang Liu et.al.|[2412.04465](http://arxiv.org/abs/2412.04465)|null|
|**2024-12-08**|**Discriminative Fine-tuning of LVLMs**|Yassine Ouali et.al.|[2412.04378](http://arxiv.org/abs/2412.04378)|null|
|**2024-12-05**|**Customize Segment Anything Model for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts**|Chenyang Zhu et.al.|[2412.04220](http://arxiv.org/abs/2412.04220)|null|
|**2024-12-05**|**SoRA: Singular Value Decomposed Low-Rank Adaptation for Domain Generalizable Representation Learning**|Seokju Yun et.al.|[2412.04077](http://arxiv.org/abs/2412.04077)|**[link](https://github.com/ysj9909/DG-SoRA)**|
|**2024-12-04**|**Personalizing Multimodal Large Language Models for Image Captioning: An Experimental Analysis**|Davide Bucciarelli et.al.|[2412.03665](http://arxiv.org/abs/2412.03665)|null|
|**2024-12-04**|**Imagine360: Immersive 360 Video Generation from Perspective Anchor**|Jing Tan et.al.|[2412.03552](http://arxiv.org/abs/2412.03552)|null|
|**2024-12-04**|**DIVE: Taming DINO for Subject-Driven Video Editing**|Yi Huang et.al.|[2412.03347](http://arxiv.org/abs/2412.03347)|null|
|**2024-12-04**|**Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach**|Lingchen Sun et.al.|[2412.03017](http://arxiv.org/abs/2412.03017)|**[link](https://github.com/csslc/pisa-sr)**|
|**2024-12-03**|**EvRT-DETR: The Surprising Effectiveness of DETR-based Detection for Event Cameras**|Dmitrii Torbunov et.al.|[2412.02890](http://arxiv.org/abs/2412.02890)|null|
|**2024-12-03**|**Explainable CTR Prediction via LLM Reasoning**|Xiaohan Yu et.al.|[2412.02588](http://arxiv.org/abs/2412.02588)|null|
|**2024-12-03**|**LoRA Diffusion: Zero-Shot LoRA Synthesis for Diffusion Model Personalization**|Ethan Smith et.al.|[2412.02352](http://arxiv.org/abs/2412.02352)|null|
|**2024-12-03**|**SimuScope: Realistic Endoscopic Synthetic Dataset Generation through Surgical Simulation and Diffusion Models**|Sabina Martyniak et.al.|[2412.02332](http://arxiv.org/abs/2412.02332)|**[link](https://github.com/sanoscience/simuscope)**|
|**2024-12-03**|**Unlocking Tuning-Free Few-Shot Adaptability in Visual Foundation Models by Recycling Pre-Tuned LoRAs**|Zixuan Hu et.al.|[2412.02220](http://arxiv.org/abs/2412.02220)|null|
|**2024-12-02**|**Optimizing LoRa for Edge Computing with TinyML Pipeline for Channel Hopping**|Marla Grunewald et.al.|[2412.01609](http://arxiv.org/abs/2412.01609)|null|
|**2024-12-02**|**CellSeg1: Robust Cell Segmentation with One Training Image**|Peilin Zhou et.al.|[2412.01410](http://arxiv.org/abs/2412.01410)|**[link](https://github.com/Nuisal/cellseg1)**|
|**2024-12-02**|**Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking**|Marco Federici et.al.|[2412.01380](http://arxiv.org/abs/2412.01380)|null|
|**2024-12-02**|**MuLan: Adapting Multilingual Diffusion Models for Hundreds of Languages with Negligible Cost**|Sen Xing et.al.|[2412.01271](http://arxiv.org/abs/2412.01271)|null|
|**2024-12-02**|**RILQ: Rank-Insensitive LoRA-based Quantization Error Compensation for Boosting 2-bit Large Language Model Accuracy**|Geonho Lee et.al.|[2412.01129](http://arxiv.org/abs/2412.01129)|null|
|**2024-12-03**|**Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA**|Haodong Lu et.al.|[2412.01004](http://arxiv.org/abs/2412.01004)|null|
|**2024-11-29**|**SURE-VQA: Systematic Understanding of Robustness Evaluation in Medical VQA Tasks**|Kim-Celine Kahl et.al.|[2411.19688](http://arxiv.org/abs/2411.19688)|**[link](https://github.com/iml-dkfz/sure-vqa)**|
|**2024-11-29**|**Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning**|Kaustubh Ponkshe et.al.|[2411.19557](http://arxiv.org/abs/2411.19557)|**[link](https://github.com/raghavsinghal10/lora-sb)**|
|**2024-11-28**|**PEFT-as-an-Attack! Jailbreaking Language Models during Federated Parameter-Efficient Fine-Tuning**|Shenghui Li et.al.|[2411.19335](http://arxiv.org/abs/2411.19335)|null|
|**2024-11-28**|**Enhancing Parameter-Efficient Fine-Tuning of Vision Transformers through Frequency-Based Adaptation**|Son Thai Ly et.al.|[2411.19297](http://arxiv.org/abs/2411.19297)|**[link](https://github.com/tsly123/freqfit)**|
|**2024-11-28**|**LoRA of Change: Learning to Generate LoRA for the Editing Instruction from A Single Before-After Image Pair**|Xue Song et.al.|[2411.19156](http://arxiv.org/abs/2411.19156)|null|
|**2024-11-28**|**DESIRE: Dynamic Knowledge Consolidation for Rehearsal-Free Continual Learning**|Haiyang Guo et.al.|[2411.19154](http://arxiv.org/abs/2411.19154)|null|
|**2024-11-28**|**Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures**|Yicheng Zhang et.al.|[2411.19128](http://arxiv.org/abs/2411.19128)|**[link](https://github.com/zyc140345/fedamole)**|
|**2024-11-27**|**Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning**|Omkar Khade et.al.|[2411.18571](http://arxiv.org/abs/2411.18571)|null|
|**2024-11-27**|**Emergence of Self-Identity in AI: A Mathematical Framework and Empirical Study with Generative Large Language Models**|Minhyeok Lee et.al.|[2411.18530](http://arxiv.org/abs/2411.18530)|**[link](https://github.com/BrainJellyPie/self)**|
|**2024-11-27**|**Adaptive Blind All-in-One Image Restoration**|David Serrano-Lozano et.al.|[2411.18412](http://arxiv.org/abs/2411.18412)|**[link](https://github.com/davidserra9/abair)**|
|**2024-11-27**|**Thai Financial Domain Adaptation of THaLLE -- Technical Report**|KBTG Labs et.al.|[2411.18242](http://arxiv.org/abs/2411.18242)|null|
|**2024-11-27**|**ROICtrl: Boosting Instance Control for Visual Generation**|Yuchao Gu et.al.|[2411.17949](http://arxiv.org/abs/2411.17949)|null|
|**2024-11-26**|**Pretrained LLM Adapted with LoRA as a Decision Transformer for Offline RL in Quantitative Trading**|Suyeol Yun et.al.|[2411.17900](http://arxiv.org/abs/2411.17900)|**[link](https://github.com/syyunn/finrl-dt)**|
|**2024-11-26**|**Low-rank Adaptation-based All-Weather Removal for Autonomous Navigation**|Sudarshan Rajagopalan et.al.|[2411.17814](http://arxiv.org/abs/2411.17814)|null|
|**2024-11-26**|**PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**|Zhen Sun et.al.|[2411.17453](http://arxiv.org/abs/2411.17453)|null|
|**2024-11-26**|**CLOVER: Constrained Learning with Orthonormal Vectors for Eliminating Redundancy**|Fanxu Meng et.al.|[2411.17426](http://arxiv.org/abs/2411.17426)|null|
|**2024-11-26**|**Efficient Deployment of Transformer Models in Analog In-Memory Computing Hardware**|Chen Li et.al.|[2411.17367](http://arxiv.org/abs/2411.17367)|**[link](https://github.com/chenlicodebank/lora_on_analog_hardware)**|
|**2024-11-26**|**ThreatModeling-LLM: Automating Threat Modeling using Large Language Models for Banking System**|Shuiqiao Yang et.al.|[2411.17058](http://arxiv.org/abs/2411.17058)|null|
|**2024-11-26**|**PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation**|Hengjia Li et.al.|[2411.17048](http://arxiv.org/abs/2411.17048)|null|
|**2024-11-25**|**RECAST: Reparameterized, Compact weight Adaptation for Sequential Tasks**|Nazia Tasnim et.al.|[2411.16870](http://arxiv.org/abs/2411.16870)|null|
|**2024-11-25**|**Parameter Efficient Instruction Tuning: An Empirical Study**|Pengfei He et.al.|[2411.16775](http://arxiv.org/abs/2411.16775)|null|
|**2024-11-23**|**LoBAM: LoRA-Based Backdoor Attack on Model Merging**|Ming Yin et.al.|[2411.16746](http://arxiv.org/abs/2411.16746)|null|
|**2024-11-24**|**Modality Alignment Meets Federated Broadcasting**|Yuting Ma et.al.|[2411.15837](http://arxiv.org/abs/2411.15837)|null|
|**2024-11-24**|**LoRA-Mini : Adaptation Matrices Decomposition and Selective Training**|Ayush Singh et.al.|[2411.15804](http://arxiv.org/abs/2411.15804)|null|
|**2024-11-23**|**Reassessing Layer Pruning in LLMs: New Insights and Methods**|Yao Lu et.al.|[2411.15558](http://arxiv.org/abs/2411.15558)|**[link](https://github.com/yaolu-zjut/Navigation-LLM-layer-pruning)**|
|**2024-11-23**|**Gradient dynamics for low-rank fine-tuning beyond kernels**|Arif Kerem Dayi et.al.|[2411.15385](http://arxiv.org/abs/2411.15385)|null|
|**2024-11-22**|**On the Impact of Fine-Tuning on Chain-of-Thought Reasoning**|Elita Lobo et.al.|[2411.15382](http://arxiv.org/abs/2411.15382)|null|
|**2024-11-22**|**ElastiFormer: Learned Redundancy Reduction in Transformer via Self-Distillation**|Junzhang Liu et.al.|[2411.15281](http://arxiv.org/abs/2411.15281)|null|
|**2024-11-21**|**IterIS: Iterative Inference-Solving Alignment for LoRA Merging**|Hongxu Chen et.al.|[2411.15231](http://arxiv.org/abs/2411.15231)|null|
|**2024-11-22**|**Exploring Foundation Models Fine-Tuning for Cytology Classification**|Manon Dausort et.al.|[2411.14975](http://arxiv.org/abs/2411.14975)|**[link](https://github.com/mdausort/Cytology-fine-tuning)**|
|**2024-11-22**|**LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement**|Jieming Bian et.al.|[2411.14961](http://arxiv.org/abs/2411.14961)|null|
|**2024-11-21**|**Interpreting seasonal and interannual Hadley cell descending edge migrations via the cell-mean Rossby number**|Spencer A Hill et.al.|[2411.14544](http://arxiv.org/abs/2411.14544)|null|
|**2024-11-21**|**Multi LoRA Meets Vision: Merging multiple adapters to create a multi task model**|Ege Kesim et.al.|[2411.14064](http://arxiv.org/abs/2411.14064)|null|
|**2024-11-21**|**Separable Mixture of Low-Rank Adaptation for Continual Visual Instruction Tuning**|Ziqi Wang et.al.|[2411.13949](http://arxiv.org/abs/2411.13949)|null|
|**2024-11-21**|**Dressing the Imagination: A Dataset for AI-Powered Translation of Text into Fashion Outfits and A Novel KAN Adapter for Enhanced Feature Adaptation**|Gayatri Deshmukh et.al.|[2411.13901](http://arxiv.org/abs/2411.13901)|null|
|**2024-11-21**|**AutoMixQ: Self-Adjusting Quantization for High Performance Memory-Efficient Fine-Tuning**|Changhai Zhou et.al.|[2411.13814](http://arxiv.org/abs/2411.13814)|null|
|**2024-11-20**|**Unleashing the Power of Large Language Models for Group POI Recommendations**|Jing Long et.al.|[2411.13415](http://arxiv.org/abs/2411.13415)|null|
|**2024-11-20**|**On the Way to LLM Personalization: Learning to Remember User Conversations**|Lucie Charlotte Magister et.al.|[2411.13405](http://arxiv.org/abs/2411.13405)|null|
|**2024-11-19**|**Visual Cue Enhancement and Dual Low-Rank Adaptation for Efficient Visual Instruction Fine-Tuning**|Pengkun Jiao et.al.|[2411.12787](http://arxiv.org/abs/2411.12787)|null|
|**2024-11-16**|**LoRA Unlearns More and Retains More (Student Abstract)**|Atharv Mittal et.al.|[2411.11907](http://arxiv.org/abs/2411.11907)|**[link](https://github.com/vlgiitr/lora-unlearn)**|
|**2024-11-18**|**SeqProFT: Applying LoRA Finetuning for Sequence-only Protein Property Predictions**|Shuo Zhang et.al.|[2411.11530](http://arxiv.org/abs/2411.11530)|null|
|**2024-11-16**|**Awaker2.5-VL: Stably Scaling MLLMs with Parameter-Efficient Mixture of Experts**|Jinqiang Long et.al.|[2411.10669](http://arxiv.org/abs/2411.10669)|**[link](https://github.com/metabrainagi/awaker)**|
|**2024-11-15**|**AmoebaLLM: Constructing Any-Shape Large Language Models for Efficient and Instant Deployment**|Yonggan Fu et.al.|[2411.10606](http://arxiv.org/abs/2411.10606)|**[link](https://github.com/GATECH-EIC/AmoebaLLM)**|
|**2024-11-15**|**Towards Multi-View Consistent Style Transfer with One-Step Diffusion via Vision Conditioning**|Yushen Zuo et.al.|[2411.10130](http://arxiv.org/abs/2411.10130)|null|
|**2024-11-15**|**LoRA-LiteE: A Computationally Efficient Framework for Chatbot Preference-Tuning**|Yahe Yang et.al.|[2411.09947](http://arxiv.org/abs/2411.09947)|null|
|**2024-11-12**|**Structured Pattern Expansion with Diffusion Models**|Marzia Riso et.al.|[2411.08930](http://arxiv.org/abs/2411.08930)|null|
|**2024-11-13**|**Dynamic Subset Tuning: Expanding the Operational Range of Parameter-Efficient Training for Large Language Models**|Felix Stahlberg et.al.|[2411.08610](http://arxiv.org/abs/2411.08610)|null|
|**2024-11-13**|**Machine Unlearning on Pre-trained Models by Residual Feature Alignment Using LoRA**|Laiqiao Qin et.al.|[2411.08443](http://arxiv.org/abs/2411.08443)|null|
|**2024-11-11**|**LoRA-BERT: a Natural Language Processing Model for Robust and Accurate Prediction of long non-coding RNAs**|Nicholas Jeon et.al.|[2411.08073](http://arxiv.org/abs/2411.08073)|null|
|**2024-11-12**|**FRUGAL: Memory-Efficient Optimization by Reducing State Overhead for Scalable Training**|Philip Zmushko et.al.|[2411.07837](http://arxiv.org/abs/2411.07837)|**[link](https://github.com/fzmushko/frugal)**|
|**2024-11-12**|**Efficient Federated Finetuning of Tiny Transformers with Resource-Constrained Devices**|Kilian Pfeiffer et.al.|[2411.07826](http://arxiv.org/abs/2411.07826)|null|
|**2024-11-12**|**Federated Low-Rank Adaptation with Differential Privacy over Wireless Networks**|Tianqu Kang et.al.|[2411.07806](http://arxiv.org/abs/2411.07806)|null|
|**2024-11-12**|**ASER: Activation Smoothing and Error Reconstruction for Large Language Model Quantization**|Weibo Zhao et.al.|[2411.07762](http://arxiv.org/abs/2411.07762)|null|
|**2024-11-11**|**DeepONet as a Multi-Operator Extrapolation Model: Distributed Pretraining with Physics-Informed Fine-Tuning**|Zecheng Zhang et.al.|[2411.07239](http://arxiv.org/abs/2411.07239)|null|
|**2024-11-11**|**Invar-RAG: Invariant LLM-aligned Retrieval for Better Generation**|Ziwei Liu et.al.|[2411.07021](http://arxiv.org/abs/2411.07021)|null|
|**2024-11-11**|**MapSAM: Adapting Segment Anything Model for Automated Feature Detection in Historical Maps**|Xue Xia et.al.|[2411.06971](http://arxiv.org/abs/2411.06971)|null|
|**2024-11-11**|**LLM-Neo: Parameter Efficient Knowledge Distillation for Large Language Models**|Runming Yang et.al.|[2411.06839](http://arxiv.org/abs/2411.06839)|null|
|**2024-11-10**|**Federated LLMs Fine-tuned with Adaptive Importance-Aware LoRA**|Yang Su et.al.|[2411.06581](http://arxiv.org/abs/2411.06581)|null|
|**2024-11-10**|**Prompt-Efficient Fine-Tuning for GPT-like Deep Models to Reduce Hallucination and to Improve Reproducibility in Scientific Text Generation Using Stochastic Optimisation Techniques**|Daniil Sulimov et.al.|[2411.06445](http://arxiv.org/abs/2411.06445)|null|
|**2024-11-08**|**Energy Efficient Protein Language Models: Leveraging Small Language Models with LoRA for Controllable Protein Generation**|Aayush Shah et.al.|[2411.05966](http://arxiv.org/abs/2411.05966)|null|
|**2024-11-08**|**Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation**|Xiwen Wei et.al.|[2411.05663](http://arxiv.org/abs/2411.05663)|**[link](https://github.com/christina200/online-lora-official)**|
|**2024-11-08**|**SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**|Muyang Li et.al.|[2411.05007](http://arxiv.org/abs/2411.05007)|**[link](https://github.com/mit-han-lab/deepcompressor)**|
|**2024-11-07**|**DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion**|Wenqiang Sun et.al.|[2411.04928](http://arxiv.org/abs/2411.04928)|null|
|**2024-11-07**|**StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration**|Panwen Hu et.al.|[2411.04925](http://arxiv.org/abs/2411.04925)|null|
|**2024-11-07**|**LLM-R: A Framework for Domain-Adaptive Maintenance Scheme Generation Combining Hierarchical Agents and RAG**|Laifa Tao et.al.|[2411.04476](http://arxiv.org/abs/2411.04476)|null|
|**2024-11-09**|**Variational Low-Rank Adaptation Using IVON**|Bai Cong et.al.|[2411.04421](http://arxiv.org/abs/2411.04421)|**[link](https://github.com/team-approx-bayes/ivon-lora)**|
|**2024-11-08**|**Robust and Efficient Fine-tuning of LLMs with Bayesian Reparameterization of Low-Rank Adaptation**|Ayan Sengupta et.al.|[2411.04358](http://arxiv.org/abs/2411.04358)|**[link](https://github.com/lcs2-iiitd/monteclora)**|
|**2024-11-06**|**PyroGuardian: An IoT-Enabled System for Health and Location Monitoring in High-Risk Firefighting Environments**|Berkay Kaplan et.al.|[2411.03654](http://arxiv.org/abs/2411.03654)|null|
|**2024-11-05**|**LLM-based Framework for Bearing Fault Diagnosis**|Laifa Tao et.al.|[2411.02718](http://arxiv.org/abs/2411.02718)|null|
|**2024-11-04**|**TeleOracle: Fine-Tuned Retrieval-Augmented Generation with Long-Context Support for Network**|Nouf Alabbasi et.al.|[2411.02617](http://arxiv.org/abs/2411.02617)|**[link](https://github.com/Nouf-Alabbasi/oKUmura_AI_Telecom_challenge)**|
|**2024-11-04**|**Parameter-Efficient Fine-Tuning of Large Language Models for Unit Test Generation: An Empirical Study**|André Storhaug et.al.|[2411.02462](http://arxiv.org/abs/2411.02462)|null|
|**2024-11-04**|**Expanding Sparse Tuning for Low Memory Usage**|Shufan Shen et.al.|[2411.01800](http://arxiv.org/abs/2411.01800)|**[link](https://github.com/ssfgunner/snell)**|
|**2024-11-02**|**PMoL: Parameter Efficient MoE for Preference Mixing of LLM Alignment**|Dongxu Liu et.al.|[2411.01245](http://arxiv.org/abs/2411.01245)|null|
|**2024-11-02**|**One Arrow, Many Targets: Probing LLMs for Multi-Attribute Controllable Text Summarization**|Tathagato Roy et.al.|[2411.01213](http://arxiv.org/abs/2411.01213)|null|
|**2024-11-02**|**Hollowed Net for On-Device Personalization of Text-to-Image Diffusion Models**|Wonguk Cho et.al.|[2411.01179](http://arxiv.org/abs/2411.01179)|null|
|**2024-11-02**|**LoRA-Contextualizing Adaptation of Large Multimodal Models for Long Document Understanding**|Jian Chen et.al.|[2411.01106](http://arxiv.org/abs/2411.01106)|null|
|**2024-11-01**|**V-LoRA: An Efficient and Flexible System Boosts Vision Applications with LoRA LMM**|Liang Mi et.al.|[2411.00915](http://arxiv.org/abs/2411.00915)|null|
|**2024-11-01**|**Dual Low-Rank Adaptation for Continual Learning with Pre-Trained Models**|Huancheng Chen et.al.|[2411.00623](http://arxiv.org/abs/2411.00623)|null|
|**2024-10-31**|**DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion**|Weicai Ye et.al.|[2410.24203](http://arxiv.org/abs/2410.24203)|**[link](https://github.com/zju3dv/diffpano)**|
|**2024-11-05**|**In-Context LoRA for Diffusion Transformers**|Lianghua Huang et.al.|[2410.23775](http://arxiv.org/abs/2410.23775)|**[link](https://github.com/ali-vilab/In-Context-LoRA)**|
|**2024-10-30**|**Model-free Low-Rank Reinforcement Learning via Leveraged Entry-wise Matrix Estimation**|Stefan Stojanovic et.al.|[2410.23434](http://arxiv.org/abs/2410.23434)|null|
|**2024-10-31**|**SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation**|Yining Hong et.al.|[2410.23277](http://arxiv.org/abs/2410.23277)|null|
|**2024-10-31**|**Why Gradient Subspace? Identifying and Mitigating LoRA's Bottlenecks in Federated Fine-Tuning of Large Language Models**|Navyansh Mahla et.al.|[2410.23111](http://arxiv.org/abs/2410.23111)|null|
|**2024-10-30**|**Efficient Adaptation of Pre-trained Vision Transformer via Householder Transformation**|Wei Dong et.al.|[2410.22952](http://arxiv.org/abs/2410.22952)|null|
|**2024-10-30**|**CopRA: A Progressive LoRA Training Strategy**|Zhan Zhuang et.al.|[2410.22911](http://arxiv.org/abs/2410.22911)|null|
|**2024-10-30**|**Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients**|Jabin Koo et.al.|[2410.22815](http://arxiv.org/abs/2410.22815)|null|
|**2024-10-30**|**MALoRA: Mixture of Asymmetric Low-Rank Adaptation for Enhanced Multi-Task Learning**|Xujia Wang et.al.|[2410.22782](http://arxiv.org/abs/2410.22782)|null|
|**2024-10-29**|**Meta-Learning Adaptable Foundation Models**|Jacob L. Block et.al.|[2410.22264](http://arxiv.org/abs/2410.22264)|null|
|**2024-10-30**|**IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models**|Hang Guo et.al.|[2410.21759](http://arxiv.org/abs/2410.21759)|**[link](https://github.com/csguoh/intlora)**|
|**2024-10-28**|**LoRA vs Full Fine-tuning: An Illusion of Equivalence**|Reece Shuttleworth et.al.|[2410.21228](http://arxiv.org/abs/2410.21228)|null|
|**2024-10-28**|**Skip2-LoRA: A Lightweight On-device DNN Fine-tuning Method for Low-cost Edge Devices**|Hiroki Matsutani et.al.|[2410.21073](http://arxiv.org/abs/2410.21073)|null|
|**2024-10-28**|**KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation**|Rambod Azimi et.al.|[2410.20777](http://arxiv.org/abs/2410.20777)|**[link](https://github.com/rambodazimi/kd-lora)**|
|**2024-10-28**|**Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA**|Sangmin Bae et.al.|[2410.20672](http://arxiv.org/abs/2410.20672)|null|
|**2024-10-28**|**PepDoRA: A Unified Peptide Language Model via Weight-Decomposed Low-Rank Adaptation**|Leyao Wang et.al.|[2410.20667](http://arxiv.org/abs/2410.20667)|null|
|**2024-10-28**|**Collaborative Knowledge Fusion: A Novel Approach for Multi-task Recommender Systems via LLMs**|Chuang Zhao et.al.|[2410.20642](http://arxiv.org/abs/2410.20642)|null|
|**2024-10-27**|**LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization**|Jui-Nan Yen et.al.|[2410.20625](http://arxiv.org/abs/2410.20625)|null|
|**2024-10-27**|**FoldMark: Protecting Protein Generative Models with Watermarking**|Zaixi Zhang et.al.|[2410.20354](http://arxiv.org/abs/2410.20354)|**[link](https://github.com/zaixizhang/FoldMark)**|
|**2024-10-26**|**An Efficient Watermarking Method for Latent Diffusion Models via Low-Rank Adaptation**|Dongdong Lin et.al.|[2410.20202](http://arxiv.org/abs/2410.20202)|null|
|**2024-10-25**|**Model merging with SVD to tie the Knots**|George Stoica et.al.|[2410.19735](http://arxiv.org/abs/2410.19735)|**[link](https://github.com/gstoica27/knots)**|
|**2024-10-25**|**Less is More: Extreme Gradient Boost Rank-1 Adaption for Efficient Finetuning of LLMs**|Yifei Zhang et.al.|[2410.19694](http://arxiv.org/abs/2410.19694)|null|
|**2024-10-25**|**GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing**|Hosam Elgendy et.al.|[2410.19552](http://arxiv.org/abs/2410.19552)|**[link](https://github.com/HosamGen/GeoLLaVA)**|
|**2024-10-24**|**Tailored-LLaMA: Optimizing Few-Shot Learning in Pruned LLaMA Models with Task-Specific Prompts**|Danyal Aftab et.al.|[2410.19185](http://arxiv.org/abs/2410.19185)|null|
|**2024-10-24**|**On the Crucial Role of Initialization for Matrix Factorization**|Bingcong Li et.al.|[2410.18965](http://arxiv.org/abs/2410.18965)|null|
|**2024-10-24**|**PSY: Posterior Sampling Based Privacy Enhancer in Large Language Models**|Yulian Sun et.al.|[2410.18824](http://arxiv.org/abs/2410.18824)|null|
|**2024-10-24**|**GeoLoRA: Geometric integration for parameter efficient fine-tuning**|Steffen Schotthöfer et.al.|[2410.18720](http://arxiv.org/abs/2410.18720)|null|
|**2024-10-24**|**Ali-AUG: Innovative Approaches to Labeled Data Augmentation using One-Step Diffusion Model**|Ali Hamza et.al.|[2410.18678](http://arxiv.org/abs/2410.18678)|null|
|**2024-10-23**|**CLEAR: Character Unlearning in Textual and Visual Modalities**|Alexey Dontsov et.al.|[2410.18057](http://arxiv.org/abs/2410.18057)|null|
|**2024-10-23**|**MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning**|Jingfan Zhang et.al.|[2410.18035](http://arxiv.org/abs/2410.18035)|null|
|**2024-10-23**|**Closed-form merging of parameter-efficient modules for Federated Continual Learning**|Riccardo Salami et.al.|[2410.17961](http://arxiv.org/abs/2410.17961)|null|
|**2024-10-23**|**AdaRankGrad: Adaptive Gradient-Rank and Moments for Memory-Efficient LLMs Training and Fine-Tuning**|Yehonathan Refael et.al.|[2410.17881](http://arxiv.org/abs/2410.17881)|null|
|**2024-10-23**|**Understanding Layer Significance in LLM Alignment**|Guangyuan Shi et.al.|[2410.17875](http://arxiv.org/abs/2410.17875)|null|
|**2024-10-23**|**VoiceTextBlender: Augmenting Large Language Models with Speech Capabilities via Single-Stage Joint Speech-Text Supervised Fine-Tuning**|Yifan Peng et.al.|[2410.17485](http://arxiv.org/abs/2410.17485)|null|
|**2024-10-22**|**FairLoRA: Unpacking Bias Mitigation in Vision Models with Fairness-Driven Low-Rank Adaptation**|Rohan Sukumaran et.al.|[2410.17358](http://arxiv.org/abs/2410.17358)|null|
|**2024-10-22**|**Insights on Disagreement Patterns in Multimodal Safety Perception across Diverse Rater Groups**|Charvi Rastogi et.al.|[2410.17032](http://arxiv.org/abs/2410.17032)|null|
|**2024-10-23**|**GeoCode-GPT: A Large Language Model for Geospatial Code Generation Tasks**|Shuyang Hou et.al.|[2410.17031](http://arxiv.org/abs/2410.17031)|null|
|**2024-10-22**|**LoRA-C: Parameter-Efficient Fine-Tuning of Robust CNN for IoT Devices**|Chuntao Ding et.al.|[2410.16954](http://arxiv.org/abs/2410.16954)|**[link](https://github.com/alexyyds2024/lora-C)**|
|**2024-10-22**|**Can Large Language Models Act as Ensembler for Multi-GNNs?**|Hanqi Duan et.al.|[2410.16822](http://arxiv.org/abs/2410.16822)|null|
|**2024-10-22**|**Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on Large Language Models**|Yuheng Lu et.al.|[2410.16801](http://arxiv.org/abs/2410.16801)|null|
|**2024-10-22**|**MoRE: Multi-Modal Contrastive Pre-training with Transformers on X-Rays, ECGs, and Diagnostic Report**|Samrajya Thapa et.al.|[2410.16239](http://arxiv.org/abs/2410.16239)|**[link](https://github.com/svthapa/more)**|
|**2024-10-21**|**Beyond 2:4: exploring V:N:M sparsity for efficient transformer inference on GPUs**|Kang Zhao et.al.|[2410.16135](http://arxiv.org/abs/2410.16135)|null|
|**2024-10-21**|**Natural GaLore: Accelerating GaLore for memory-efficient LLM Training and Fine-tuning**|Arijit Das et.al.|[2410.16029](http://arxiv.org/abs/2410.16029)|**[link](https://github.com/selfsupervised-ai/natural-galore)**|
|**2024-10-21**|**How to Build a Pre-trained Multimodal model for Simultaneously Chatting and Decision-making?**|Zuojin Tang et.al.|[2410.15885](http://arxiv.org/abs/2410.15885)|null|
|**2024-10-21**|**The effect of fine-tuning on language model toxicity**|Will Hawkins et.al.|[2410.15821](http://arxiv.org/abs/2410.15821)|**[link](https://github.com/willhawkins3/finetuningtoxicity)**|
|**2024-10-21**|**Habaek: High-performance water segmentation through dataset expansion and inductive bias optimization**|Hanseon Joo et.al.|[2410.15794](http://arxiv.org/abs/2410.15794)|**[link](https://github.com/HanseonJoo/Habaek)**|
|**2024-10-21**|**Students Rather Than Experts: A New AI For Education Pipeline To Model More Human-Like And Personalised Early Adolescences**|Yiping Ma et.al.|[2410.15701](http://arxiv.org/abs/2410.15701)|null|
|**2024-10-20**|**MIRA: A Method of Federated MultI-Task Learning for LaRge LAnguage Models**|Ahmed Elbakary et.al.|[2410.15524](http://arxiv.org/abs/2410.15524)|null|
|**2024-10-20**|**EVA: An Embodied World Model for Future Video Anticipation**|Xiaowei Chi et.al.|[2410.15461](http://arxiv.org/abs/2410.15461)|null|
|**2024-10-20**|**LoRA-IR: Taming Low-Rank Experts for Efficient All-in-One Image Restoration**|Yuang Ai et.al.|[2410.15385](http://arxiv.org/abs/2410.15385)|**[link](https://github.com/shallowdream204/lora-ir)**|
|**2024-10-18**|**Fine-Tuning DeepONets to Enhance Physics-informed Neural Networks for solving Partial Differential Equations**|Sidi Wu et.al.|[2410.14134](http://arxiv.org/abs/2410.14134)|null|
|**2024-10-17**|**FiTv2: Scalable and Improved Flexible Vision Transformer for Diffusion Model**|ZiDong Wang et.al.|[2410.13925](http://arxiv.org/abs/2410.13925)|**[link](https://github.com/whlzy/fit)**|
|**2024-10-17**|**Improving Multi-modal Large Language Model through Boosting Vision Capabilities**|Yanpeng Sun et.al.|[2410.13733](http://arxiv.org/abs/2410.13733)|null|
|**2024-10-17**|**LoLDU: Low-Rank Adaptation via Lower-Diag-Upper Decomposition for Parameter-Efficient Fine-Tuning**|Yiming Shi et.al.|[2410.13618](http://arxiv.org/abs/2410.13618)|**[link](https://github.com/skddj/loldu)**|
|**2024-10-18**|**MoR: Mixture of Ranks for Low-Rank Adaptation Tuning**|Chuanyu Tang et.al.|[2410.13408](http://arxiv.org/abs/2410.13408)|null|
|**2024-10-17**|**FAMSeC: A Few-shot-sample-based General AI-generated Image Detection Method**|Juncong Xu et.al.|[2410.13156](http://arxiv.org/abs/2410.13156)|null|
|**2024-10-16**|**LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks**|Akshara Prabhakar et.al.|[2410.13025](http://arxiv.org/abs/2410.13025)|**[link](https://github.com/aksh555/lora-soups)**|
|**2024-10-16**|**DEeR: Deviation Eliminating and Noise Regulating for Privacy-preserving Federated Low-rank Adaptation**|Meilu Zhu et.al.|[2410.12926](http://arxiv.org/abs/2410.12926)|**[link](https://github.com/cuhk-aim-group/deer)**|
|**2024-10-15**|**In-context KV-Cache Eviction for LLMs via Attention-Gate**|Zihao Zeng et.al.|[2410.12876](http://arxiv.org/abs/2410.12876)|null|
|**2024-10-16**|**FiRST: Finetuning Router-Selective Transformers for Input-Adaptive Latency Reduction**|Akriti Jain et.al.|[2410.12513](http://arxiv.org/abs/2410.12513)|null|
|**2024-10-15**|**LoKO: Low-Rank Kalman Optimizer for Online Fine-Tuning of Large Models**|Hossein Abdi et.al.|[2410.11551](http://arxiv.org/abs/2410.11551)|null|
|**2024-10-15**|**Transfer Learning with Foundational Models for Time Series Forecasting using Low-Rank Adaptations**|M. Germán-Morales et.al.|[2410.11539](http://arxiv.org/abs/2410.11539)|null|
|**2024-10-15**|**Energy Efficient Transmission Parameters Selection Method Using Reinforcement Learning in Distributed LoRa Networks**|Ryotai Airiyoshi et.al.|[2410.11270](http://arxiv.org/abs/2410.11270)|null|
|**2024-10-14**|**Improving the Language Understanding Capabilities of Large Language Models Using Reinforcement Learning**|Bokai Hu et.al.|[2410.11020](http://arxiv.org/abs/2410.11020)|null|
|**2024-10-14**|**LoLCATs: On Low-Rank Linearizing of Large Language Models**|Michael Zhang et.al.|[2410.10254](http://arxiv.org/abs/2410.10254)|**[link](https://github.com/hazyresearch/lolcats)**|
|**2024-10-14**|**Fed-piLot: Optimizing LoRA Assignment for Efficient Federated Foundation Model Fine-Tuning**|Zikai Zhang et.al.|[2410.10200](http://arxiv.org/abs/2410.10200)|null|
|**2024-10-14**|**Scalable Multi-Domain Adaptation of Language Models using Modular Experts**|Peter Schafhalter et.al.|[2410.10181](http://arxiv.org/abs/2410.10181)|null|
|**2024-10-14**|**Is Parameter Collision Hindering Continual Learning in LLMs?**|Shuo Yang et.al.|[2410.10179](http://arxiv.org/abs/2410.10179)|**[link](https://github.com/pku-yuangroup/n-lora)**|
|**2024-10-14**|**AlphaLoRA: Assigning LoRA Experts Based on Layer Training Quality**|Peijun Qing et.al.|[2410.10054](http://arxiv.org/abs/2410.10054)|**[link](https://github.com/morelife2017/alphalora)**|
|**2024-10-13**|**Retrieval Instead of Fine-tuning: A Retrieval-based Parameter Ensemble for Zero-shot Learning**|Pengfei Jin et.al.|[2410.09908](http://arxiv.org/abs/2410.09908)|null|
|**2024-10-13**|**A Quantum Circuit-Based Compression Perspective for Parameter-Efficient Learning**|Chen-Yu Liu et.al.|[2410.09846](http://arxiv.org/abs/2410.09846)|null|
|**2024-10-13**|**Understanding Robustness of Parameter-Efficient Tuning for Image Classification**|Jiacheng Ruan et.al.|[2410.09845](http://arxiv.org/abs/2410.09845)|**[link](https://github.com/jcruan519/petrobustness)**|
|**2024-10-13**|**BiDoRA: Bi-level Optimization-Based Weight-Decomposed Low-Rank Adaptation**|Peijia Qin et.al.|[2410.09758](http://arxiv.org/abs/2410.09758)|null|
|**2024-10-13**|**AM-SAM: Automated Prompting and Mask Calibration for Segment Anything Model**|Yuchen Li et.al.|[2410.09714](http://arxiv.org/abs/2410.09714)|null|
|**2024-10-11**|**Parameter-Efficient Fine-Tuning of State Space Models**|Kevin Galim et.al.|[2410.09016](http://arxiv.org/abs/2410.09016)|**[link](https://github.com/furiosa-ai/ssm-peft)**|
|**2024-10-10**|**Randomized Asymmetric Chain of LoRA: The First Meaningful Theoretical Framework for Low-Rank Adaptation**|Grigory Malinovsky et.al.|[2410.08305](http://arxiv.org/abs/2410.08305)|null|
|**2024-10-10**|**SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture**|Jiayi Han et.al.|[2410.07739](http://arxiv.org/abs/2410.07739)|null|
|**2024-10-10**|**MotionAura: Generating High-Quality and Motion Consistent Videos using Discrete Diffusion**|Onkar Susladkar et.al.|[2410.07659](http://arxiv.org/abs/2410.07659)|null|
|**2024-10-09**|**SparseGrad: A Selective Method for Efficient Fine-tuning of MLP Layers**|Viktoriia Chekalina et.al.|[2410.07383](http://arxiv.org/abs/2410.07383)|**[link](https://github.com/sayankotor/sparse_grads)**|
|**2024-10-09**|**One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation**|Fabian Paischer et.al.|[2410.07170](http://arxiv.org/abs/2410.07170)|**[link](https://github.com/ml-jku/EVA)**|
|**2024-10-09**|**Industrial complexity and the evolution of formal employment in developing cities**|Neave O'Clery et.al.|[2410.06971](http://arxiv.org/abs/2410.06971)|null|
|**2024-10-11**|**Enhancing Multimodal LLM for Detailed and Accurate Video Captioning using Multi-Round Preference Optimization**|Changli Tang et.al.|[2410.06682](http://arxiv.org/abs/2410.06682)|null|
|**2024-10-08**|**Systematic 2.5 D resistive MHD simulations with ambipolar diffusion and Hall effect for fast magnetic reconnection**|Gabriela Landinez et.al.|[2410.06391](http://arxiv.org/abs/2410.06391)|null|
|**2024-10-08**|**HyperDet: Generalizable Detection of Synthesized Images by Generating and Merging A Mixture of Hyper LoRAs**|Huangsen Cao et.al.|[2410.06044](http://arxiv.org/abs/2410.06044)|null|
|**2024-10-08**|**QERA: an Analytical Framework for Quantization Error Reconstruction**|Cheng Zhang et.al.|[2410.06040](http://arxiv.org/abs/2410.06040)|null|
|**2024-10-08**|**Hyper Adversarial Tuning for Boosting Adversarial Robustness of Pretrained Large Vision Models**|Kangtao Lv et.al.|[2410.05951](http://arxiv.org/abs/2410.05951)|null|
|**2024-10-07**|**GS-VTON: Controllable 3D Virtual Try-on with Gaussian Splatting**|Yukang Cao et.al.|[2410.05259](http://arxiv.org/abs/2410.05259)|null|
|**2024-10-08**|**PAMLR: A Passive-Active Multi-Armed Bandit-Based Solution for LoRa Channel Allocation**|Jihoon Yun et.al.|[2410.05147](http://arxiv.org/abs/2410.05147)|null|
|**2024-10-07**|**HyperINF: Unleashing the HyperPower of the Schulz's Method for Data Influence Estimation**|Xinyu Zhou et.al.|[2410.05090](http://arxiv.org/abs/2410.05090)|**[link](https://github.com/blackzxy/hyperinf)**|
|**2024-10-07**|**Low-Rank Continual Pyramid Vision Transformer: Incrementally Segment Whole-Body Organs in CT with Light-Weighted Adaptation**|Vince Zhu et.al.|[2410.04689](http://arxiv.org/abs/2410.04689)|null|
|**2024-10-06**|**Learning De-Biased Representations for Remote-Sensing Imagery**|Zichen Tian et.al.|[2410.04546](http://arxiv.org/abs/2410.04546)|**[link](https://github.com/doem97/deblora)**|
|**2024-10-05**|**Learning on LoRAs: GL-Equivariant Processing of Low-Rank Weight Spaces for Large Finetuned Models**|Theo et.al.|[2410.04207](http://arxiv.org/abs/2410.04207)|null|
|**2024-10-05**|**LoRTA: Low Rank Tensor Adaptation of Large Language Models**|Ignacio Hounie et.al.|[2410.04060](http://arxiv.org/abs/2410.04060)|null|
|**2024-10-05**|**Hyperbolic Fine-tuning for Large Language Models**|Menglin Yang et.al.|[2410.04010](http://arxiv.org/abs/2410.04010)|**[link](https://github.com/marlin-codes/HypLLM)**|
|**2024-10-04**|**AutoLoRA: AutoGuidance Meets Low-Rank Adaptation for Diffusion Models**|Artur Kasymov et.al.|[2410.03941](http://arxiv.org/abs/2410.03941)|**[link](https://github.com/gmum/AutoLoRA)**|
|**2024-10-04**|**Collaborative and Efficient Personalization with Mixtures of Adaptors**|Abdulla Jasem Almansoori et.al.|[2410.03497](http://arxiv.org/abs/2410.03497)|null|
|**2024-10-03**|**Neutral residues: revisiting adapters for model extension**|Franck Signe Talla et.al.|[2410.02744](http://arxiv.org/abs/2410.02744)|null|
|**2024-10-03**|**Encryption-Friendly LLM Architecture**|Donghwan Rho et.al.|[2410.02486](http://arxiv.org/abs/2410.02486)|null|
|**2024-10-02**|**NEAT: Nonlinear Parameter-efficient Adaptation of Pre-trained Models**|Yibo Zhong et.al.|[2410.01870](http://arxiv.org/abs/2410.01870)|null|
|**2024-10-02**|**Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?**|Xi Chen et.al.|[2410.01623](http://arxiv.org/abs/2410.01623)|**[link](https://github.com/xichen-fy/fira)**|
|**2024-10-02**|**DLP-LoRA: Efficient Task-Specific LoRA Fusion with a Dynamic, Lightweight Plugin for Large Language Models**|Yuxuan Zhang et.al.|[2410.01497](http://arxiv.org/abs/2410.01497)|**[link](https://github.com/mecuping/dlp-lora)**|
|**2024-10-04**|**Selective Aggregation for Low-Rank Adaptation in Federated Learning**|Pengxin Guo et.al.|[2410.01463](http://arxiv.org/abs/2410.01463)|**[link](https://github.com/Pengxin-Guo/FedSA-LoRA)**|
|**2024-10-02**|**FlashMask: Efficient and Rich Mask Extension of FlashAttention**|Guoxia Wang et.al.|[2410.01359](http://arxiv.org/abs/2410.01359)|**[link](https://github.com/PaddlePaddle/Paddle)**|
|**2024-10-01**|**MoS: Unleashing Parameter Efficiency of Low-Rank Adaptation with Mixture of Shards**|Sheng Wang et.al.|[2410.00938](http://arxiv.org/abs/2410.00938)|null|
|**2024-10-02**|**Mining Your Own Secrets: Diffusion Classifier Scores for Continual Personalization of Text-to-Image Diffusion Models**|Saurav Jha et.al.|[2410.00700](http://arxiv.org/abs/2410.00700)|null|
|**2024-10-01**|**PrivTuner with Homomorphic Encryption and LoRA: A P3EFT Scheme for Privacy-Preserving Parameter-Efficient Fine-Tuning of AI Foundation Models**|Yang Li et.al.|[2410.00433](http://arxiv.org/abs/2410.00433)|null|
|**2024-09-30**|**Fisher Information-based Efficient Curriculum Federated Learning with Large Language Models**|Ji Liu et.al.|[2410.00131](http://arxiv.org/abs/2410.00131)|null|
|**2024-09-30**|**UIR-LoRA: Achieving Universal Image Restoration through Multiple Low-Rank Adaptation**|Cheng Zhang et.al.|[2409.20197](http://arxiv.org/abs/2409.20197)|**[link](https://github.com/justones/uir-lora)**|
|**2024-09-30**|**BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain**|Kaisi Guan et.al.|[2409.20075](http://arxiv.org/abs/2409.20075)|null|
|**2024-09-30**|**HDMoLE: Mixture of LoRA Experts with Hierarchical Routing and Dynamic Thresholds for Fine-Tuning LLM-based ASR Models**|Bingshen Mu et.al.|[2409.19878](http://arxiv.org/abs/2409.19878)|null|
|**2024-09-29**|**Learning Attentional Mixture of LoRAs for Language Model Continual Learning**|Jialin Liu et.al.|[2409.19611](http://arxiv.org/abs/2409.19611)|null|
|**2024-09-29**|**Abstractive Summarization of Low resourced Nepali language using Multilingual Transformers**|Prakash Dhakal et.al.|[2409.19566](http://arxiv.org/abs/2409.19566)|null|
|**2024-09-27**|**HM3: Heterogeneous Multi-Class Model Merging**|Stefan Hackmann et.al.|[2409.19173](http://arxiv.org/abs/2409.19173)|null|
|**2024-09-26**|**MARS: Multi-radio Architecture with Radio Selection using Decision Trees for emerging mesoscale CPS/IoT applications**|Jothi Prasanna Shanmuga Sundaram et.al.|[2409.18043](http://arxiv.org/abs/2409.18043)|null|
|**2024-09-26**|**PEDRO: Parameter-Efficient Fine-tuning with Prompt DEpenDent Representation MOdification**|Tianfang Xie et.al.|[2409.17834](http://arxiv.org/abs/2409.17834)|null|
|**2024-09-30**|**Efficient In-Domain Question Answering for Resource-Constrained Environments**|Isaac Chung et.al.|[2409.17648](http://arxiv.org/abs/2409.17648)|null|
|**2024-09-26**|**On the Implicit Relation Between Low-Rank Adaptation and Differential Privacy**|Saber Malekmohammadi et.al.|[2409.17538](http://arxiv.org/abs/2409.17538)|null|
|**2024-09-26**|**A Time Series is Worth Five Experts: Heterogeneous Mixture of Experts for Traffic Flow Prediction**|Guangyu Wang et.al.|[2409.17440](http://arxiv.org/abs/2409.17440)|**[link](https://github.com/sqlcow/TITAN)**|
|**2024-09-25**|**Parameter-efficient Bayesian Neural Networks for Uncertainty-aware Depth Estimation**|Richard D. Paul et.al.|[2409.17085](http://arxiv.org/abs/2409.17085)|null|
|**2024-09-25**|**Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors**|Aiping Zhang et.al.|[2409.17058](http://arxiv.org/abs/2409.17058)|**[link](https://github.com/arctichare105/s3diff)**|
|**2024-09-25**|**PMSS: Pretrained Matrices Skeleton Selection for LLM Fine-tuning**|Qibin Wang et.al.|[2409.16722](http://arxiv.org/abs/2409.16722)|null|
|**2024-09-25**|**GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning**|Zhe-Rui Yang et.al.|[2409.16670](http://arxiv.org/abs/2409.16670)|null|
|**2024-09-25**|**Prompt Sliders for Fine-Grained Control, Editing and Erasing of Concepts in Diffusion Models**|Deepak Sridhar et.al.|[2409.16535](http://arxiv.org/abs/2409.16535)|**[link](https://github.com/deepaksridhar/promptsliders)**|
|**2024-09-24**|**Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering**|Ziyu Zhao et.al.|[2409.16167](http://arxiv.org/abs/2409.16167)|null|
|**2024-09-24**|**Evaluation of state-of-the-art ASR Models in Child-Adult Interactions**|Aditya Ashvin et.al.|[2409.16135](http://arxiv.org/abs/2409.16135)|null|
|**2024-09-24**|**Bridging Speech and Text: Enhancing ASR with Pinyin-to-Character Pre-training in LLMs**|Yang Yuhang et.al.|[2409.16005](http://arxiv.org/abs/2409.16005)|null|
|**2024-09-24**|**Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM**|Fengrun Zhang et.al.|[2409.15905](http://arxiv.org/abs/2409.15905)|null|
|**2024-09-24**|**Aided design of bridge aesthetics based on Stable Diffusion fine-tuning**|Leye Zhang et.al.|[2409.15812](http://arxiv.org/abs/2409.15812)|**[link](https://github.com/zhangleye/Bridge-SD)**|
|**2024-09-17**|**Chain-of-Thought Prompting for Speech Translation**|Ke Hu et.al.|[2409.11538](http://arxiv.org/abs/2409.11538)|null|
|**2024-09-17**|**Beyond LoRA: Exploring Efficient Fine-Tuning Techniques for Time Series Foundational Models**|Divij Gupta et.al.|[2409.11302](http://arxiv.org/abs/2409.11302)|null|
|**2024-09-17**|**LoRa Communication for Agriculture 4.0: Opportunities, Challenges, and Future Directions**|Lameya Aldhaheri et.al.|[2409.11200](http://arxiv.org/abs/2409.11200)|null|
|**2024-09-17**|**Few-Shot Domain Adaptation for Learned Image Compression**|Tianyu Zhang et.al.|[2409.11111](http://arxiv.org/abs/2409.11111)|null|
|**2024-09-17**|**KVPruner: Structural Pruning for Faster and Memory-Efficient Large Language Models**|Bo Lv et.al.|[2409.11057](http://arxiv.org/abs/2409.11057)|null|
|**2024-09-18**|**Propulsion: Steering LLM with Tiny Fine-Tuning**|Md Kowsher et.al.|[2409.10927](http://arxiv.org/abs/2409.10927)|**[link](https://github.com/Kowsher/Propulsion)**|
|**2024-09-16**|**A Bayesian Interpretation of Adaptive Low-Rank Adaptation**|Haolin Chen et.al.|[2409.10673](http://arxiv.org/abs/2409.10673)|**[link](https://github.com/idiap/vilora)**|
|**2024-09-16**|**From Text to Emoji: How PEFT-Driven Personality Manipulation Unleashes the Emoji Potential in LLMs**|Navya Jain et.al.|[2409.10245](http://arxiv.org/abs/2409.10245)|null|
|**2024-09-16**|**Robust Bird's Eye View Segmentation by Adapting DINOv2**|Merve Rabia Barın et.al.|[2409.10228](http://arxiv.org/abs/2409.10228)|null|
|**2024-09-19**|**jina-embeddings-v3: Multilingual Embeddings With Task LoRA**|Saba Sturua et.al.|[2409.10173](http://arxiv.org/abs/2409.10173)|null|
|**2024-09-16**|**Rapid Adaptation of Earth Observation Foundation Models for Segmentation**|Karthick Panner Selvam et.al.|[2409.09907](http://arxiv.org/abs/2409.09907)|null|
|**2024-09-15**|**AlpaPICO: Extraction of PICO Frames from Clinical Trial Documents Using LLMs**|Madhusudan Ghosh et.al.|[2409.09704](http://arxiv.org/abs/2409.09704)|**[link](https://github.com/shrimonmuke0202/alpapico)**|
|**2024-09-14**|**COMFORT: A Continual Fine-Tuning Framework for Foundation Models Targeted at Consumer Healthcare**|Chia-Hao Li et.al.|[2409.09549](http://arxiv.org/abs/2409.09549)|null|
|**2024-09-14**|**SAM-OCTA2: Layer Sequence OCTA Segmentation with Fine-tuned Segment Anything Model 2**|Xinrun Chen et.al.|[2409.09286](http://arxiv.org/abs/2409.09286)|**[link](https://github.com/shellredia/sam-octa2)**|
|**2024-09-13**|**Data Efficient Child-Adult Speaker Diarization with Simulated Conversations**|Anfeng Xu et.al.|[2409.08881](http://arxiv.org/abs/2409.08881)|**[link](https://github.com/usc-sail/child-adult-diarization)**|
|**2024-09-13**|**Large Language Model Can Transcribe Speech in Multi-Talker Scenarios with Versatile Instructions**|Lingwei Meng et.al.|[2409.08596](http://arxiv.org/abs/2409.08596)|null|
|**2024-09-13**|**ATFLRec: A Multimodal Recommender System with Audio-Text Fusion and Low-Rank Adaptation via Instruction-Tuned Large Language Model**|Zezheng Qin et.al.|[2409.08543](http://arxiv.org/abs/2409.08543)|null|
|**2024-09-13**|**Risks When Sharing LoRA Fine-Tuned Diffusion Model Weights**|Dixi Yao et.al.|[2409.08482](http://arxiv.org/abs/2409.08482)|null|
|**2024-09-13**|**Toward satisfactory public accessibility: A crowdsourcing approach through online reviews to inclusive urban design**|Lingyao Li et.al.|[2409.08459](http://arxiv.org/abs/2409.08459)|null|
|**2024-09-12**|**AudioBERT: Audio Knowledge Augmented Language Model**|Hyunjong Ok et.al.|[2409.08199](http://arxiv.org/abs/2409.08199)|**[link](https://github.com/hj-ok/audiobert)**|
|**2024-09-12**|**Advancing Depth Anything Model for Unsupervised Monocular Depth Estimation in Endoscopy**|Bojian Li et.al.|[2409.07723](http://arxiv.org/abs/2409.07723)|null|
|**2024-09-11**|**Efficient Localized Adaptation of Neural Weather Forecasting: A Case Study in the MENA Region**|Muhammad Akhtar Munir et.al.|[2409.07585](http://arxiv.org/abs/2409.07585)|**[link](https://github.com/akhtarvision/weather-regional)**|
|**2024-09-11**|**Improving Anomalous Sound Detection via Low-Rank Adaptation Fine-Tuning of Pre-Trained Audio Models**|Xinhu Zheng et.al.|[2409.07016](http://arxiv.org/abs/2409.07016)|null|
|**2024-09-10**|**SaRA: High-Efficient Diffusion Model Fine-tuning with Progressive Sparse Low-Rank Adaptation**|Teng Hu et.al.|[2409.06633](http://arxiv.org/abs/2409.06633)|null|
|**2024-09-09**|**Elucidating Optimal Reward-Diversity Tradeoffs in Text-to-Image Diffusion Models**|Rohit Jena et.al.|[2409.06493](http://arxiv.org/abs/2409.06493)|null|
|**2024-09-10**|**HexaCoder: Secure Code Generation via Oracle-Guided Synthetic Training Data**|Hossein Hajipour et.al.|[2409.06446](http://arxiv.org/abs/2409.06446)|**[link](https://github.com/hexacoder-ai/hexacoder)**|
|**2024-09-10**|**VE: Modeling Multivariate Time Series Correlation with Variate Embedding**|Shangjiong Wang et.al.|[2409.06169](http://arxiv.org/abs/2409.06169)|**[link](https://github.com/swang-song/ve)**|
|**2024-09-09**|**FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations**|Ziyao Wang et.al.|[2409.05976](http://arxiv.org/abs/2409.05976)|**[link](https://github.com/atp-1010/federatedllm)**|
|**2024-09-09**|**SVFit: Parameter-Efficient Fine-Tuning of Large Pre-Trained Models Using Singular Values**|Chengwei Sun et.al.|[2409.05926](http://arxiv.org/abs/2409.05926)|null|
|**2024-09-09**|**TriplePlay: Enhancing Federated Learning with CLIP for Non-IID Data and Resource Efficiency**|Ahmed Imteaj et.al.|[2409.05347](http://arxiv.org/abs/2409.05347)|null|
|**2024-09-08**|**Exploring Intrinsic Language-specific Subspaces in Fine-tuning Multilingual Neural Machine Translation**|Zhe Cao et.al.|[2409.05224](http://arxiv.org/abs/2409.05224)|**[link](https://github.com/spike0924/lslo)**|
|**2024-09-06**|**Customizing Large Language Model Generation Style using Parameter-Efficient Finetuning**|Xinyue Liu et.al.|[2409.04574](http://arxiv.org/abs/2409.04574)|null|
|**2024-09-06**|**Fast Forwarding Low-Rank Training**|Adir Rahamim et.al.|[2409.04206](http://arxiv.org/abs/2409.04206)|null|
|**2024-09-05**|**Continual Skill and Task Learning via Dialogue**|Weiwei Gu et.al.|[2409.03166](http://arxiv.org/abs/2409.03166)|null|
|**2024-09-04**|**Non-Orthogonal Multiple-Access Strategies for Direct-to-Satellite IoT Networks**|Felipe Augusto Tondo et.al.|[2409.02748](http://arxiv.org/abs/2409.02748)|null|
|**2024-09-04**|**Robust Federated Finetuning of Foundation Models via Alternating Minimization of LoRA**|Shuangyi Chen et.al.|[2409.02346](http://arxiv.org/abs/2409.02346)|null|
|**2024-08-31**|**CoRA: Optimizing Low-Rank Adaptation with Common Subspace of Large Language Models**|Xiaojun Xiao et.al.|[2409.02119](http://arxiv.org/abs/2409.02119)|null|
|**2024-09-02**|**LoGex: Improved tail detection of extremely rare histopathology classes via guided diffusion**|Maximilian Mueller et.al.|[2409.01317](http://arxiv.org/abs/2409.01317)|**[link](https://github.com/mueller-mp/logex)**|
|**2024-09-02**|**Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning**|Chongjie Si et.al.|[2409.01035](http://arxiv.org/abs/2409.01035)|**[link](https://github.com/Chongjie-Si/Subspace-Tuning)**|
|**2024-09-02**|**Personalized Lip Reading: Adapting to Your Unique Lip Movements with Vision and Language**|Jeong Hun Yeo et.al.|[2409.00986](http://arxiv.org/abs/2409.00986)|**[link](https://github.com/jeonghun0716/personalized-lip-reading)**|
|**2024-08-30**|**Enhancing Event Reasoning in Large Language Models through Instruction Fine-Tuning with Semantic Causal Graphs**|Mazal Bethany et.al.|[2409.00209](http://arxiv.org/abs/2409.00209)|null|
|**2024-08-30**|**DARES: Depth Anything in Robotic Endoscopic Surgery with Self-supervised Vector-LoRA of the Foundation Model**|Mona Sheikh Zeinoddin et.al.|[2408.17433](http://arxiv.org/abs/2408.17433)|**[link](https://github.com/mobarakol/dares)**|
|**2024-08-30**|**MoRe Fine-Tuning with 10x Fewer Parameters**|Wenxuan Tan et.al.|[2408.17383](http://arxiv.org/abs/2408.17383)|**[link](https://github.com/sprocketlab/sparse_matrix_fine_tuning)**|
|**2024-08-30**|**Wireless Integrated Authenticated Communication System (WIA-Comm)**|Amith N Bharadwaj et.al.|[2408.17112](http://arxiv.org/abs/2408.17112)|null|
|**2024-09-02**|**Instant Adversarial Purification with Adversarial Consistency Distillation**|Chun Tong Lei et.al.|[2408.17064](http://arxiv.org/abs/2408.17064)|null|
|**2024-08-30**|**Efficient Image Restoration through Low-Rank Adaptation and Stable Diffusion XL**|Haiyang Zhao et.al.|[2408.17060](http://arxiv.org/abs/2408.17060)|null|
|**2024-08-29**|**LoraMap: Harnessing the Power of LoRA Connections**|Hyeryun Park et.al.|[2408.16264](http://arxiv.org/abs/2408.16264)|null|
|**2024-08-28**|**LeMON: Learning to Learn Multi-Operator Networks**|Jingmin Sun et.al.|[2408.16168](http://arxiv.org/abs/2408.16168)|**[link](https://github.com/jingminsun/lemon_prose)**|
|**2024-08-28**|**Leveraging Open Knowledge for Advancing Task Expertise in Large Language Models**|Yuncheng Yang et.al.|[2408.15915](http://arxiv.org/abs/2408.15915)|**[link](https://github.com/yaphabates/rocket)**|
|**2024-08-28**|**StyleRemix: Interpretable Authorship Obfuscation via Distillation and Perturbation of Style Elements**|Jillian Fisher et.al.|[2408.15666](http://arxiv.org/abs/2408.15666)|**[link](https://github.com/jfisher52/StyleRemix)**|
|**2024-08-28**|**TeFF: Tracking-enhanced Forgetting-free Few-shot 3D LiDAR Semantic Segmentation**|Junbao Zhou et.al.|[2408.15657](http://arxiv.org/abs/2408.15657)|**[link](https://github.com/junbao-zhou/track-no-forgetting)**|
|**2024-08-28**|**Whisper-PMFA: Partial Multi-Scale Feature Aggregation for Speaker Verification using Whisper Models**|Yiyang Zhao et.al.|[2408.15585](http://arxiv.org/abs/2408.15585)|null|
|**2024-08-28**|**VoiceTailor: Lightweight Plug-In Adapter for Diffusion-Based Personalized Text-to-Speech**|Heeseung Kim et.al.|[2408.14739](http://arxiv.org/abs/2408.14739)|null|
|**2024-08-27**|**PAT: Pruning-Aware Tuning for Large Language Models**|Yijiang Liu et.al.|[2408.14721](http://arxiv.org/abs/2408.14721)|**[link](https://github.com/kriskrisliu/pat_pruning-aware-tuning)**|
|**2024-08-27**|**StyleSpeech: Parameter-efficient Fine Tuning for Pre-trained Controllable Text-to-Speech**|Haowei Lou et.al.|[2408.14713](http://arxiv.org/abs/2408.14713)|**[link](https://github.com/haoweilou/StyleSpeech_Public)**|
|**2024-08-26**|**CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation**|Muhammad Fawi et.al.|[2408.14572](http://arxiv.org/abs/2408.14572)|**[link](https://github.com/mnoorfawi/curlora)**|
|**2024-08-27**|**Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models**|Aradhye Agarwal et.al.|[2408.14470](http://arxiv.org/abs/2408.14470)|**[link](https://github.com/Aradhye2002/selective-peft-toolkit)**|
|**2024-08-26**|**Reprogramming Foundational Large Language Models(LLMs) for Enterprise Adoption for Spatio-Temporal Forecasting Applications: Unveiling a New Era in Copilot-Guided Cross-Modal Time Series Representation Learning**|Sakhinana Sagar Srinivas et.al.|[2408.14387](http://arxiv.org/abs/2408.14387)|null|
|**2024-08-27**|**SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher**|Trung Dao et.al.|[2408.14176](http://arxiv.org/abs/2408.14176)|**[link](https://github.com/vinairesearch/swiftbrushv2)**|
|**2024-08-25**|**TalkLoRA: Low-Rank Adaptation for Speech-Driven Animation**|Jack Saunders et.al.|[2408.13714](http://arxiv.org/abs/2408.13714)|null|
|**2024-08-24**|**Can Visual Foundation Models Achieve Long-term Point Tracking?**|Görkay Aydemir et.al.|[2408.13575](http://arxiv.org/abs/2408.13575)|null|
|**2024-08-23**|**The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities**|Venkatesh Balavadhani Parthasarathy et.al.|[2408.13296](http://arxiv.org/abs/2408.13296)|null|
|**2024-08-23**|**CLLMFS: A Contrastive Learning enhanced Large Language Model Framework for Few-Shot Named Entity Recognition**|Yafeng Zhang et.al.|[2408.12834](http://arxiv.org/abs/2408.12834)|null|
|**2024-08-23**|**Investigating LLM Applications in E-Commerce**|Chester Palen-Michel et.al.|[2408.12779](http://arxiv.org/abs/2408.12779)|null|
|**2024-08-22**|**EvalYaks: Instruction Tuning Datasets and LoRA Fine-tuned Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts**|Nicy Scaria et.al.|[2408.12226](http://arxiv.org/abs/2408.12226)|**[link](https://github.com/talking-yak/evalyaks)**|
|**2024-08-21**|**Leveraging Fine-Tuned Retrieval-Augmented Generation with Long-Context Support: For 3GPP Standards**|Omar Erak et.al.|[2408.11775](http://arxiv.org/abs/2408.11775)|**[link](https://github.com/Nouf-Alabbasi/oKUmura_AI_Telecom_challenge)**|
|**2024-08-21**|**EAGLE: Elevating Geometric Reasoning through LLM-empowered Visual Instruction Tuning**|Zhihao Li et.al.|[2408.11397](http://arxiv.org/abs/2408.11397)|null|
|**2024-08-20**|**EELE: Exploring Efficient and Extensible LoRA Integration in Emotional Text-to-Speech**|Xin Qi et.al.|[2408.10852](http://arxiv.org/abs/2408.10852)|null|
|**2024-08-21**|**Flexora: Flexible Low Rank Adaptation for Large Language Models**|Chenxing Wei et.al.|[2408.10774](http://arxiv.org/abs/2408.10774)|null|
|**2024-08-20**|**Large Language Models for Multimodal Deformable Image Registration**|Mingrui Ma et.al.|[2408.10703](http://arxiv.org/abs/2408.10703)|**[link](https://github.com/ninjannn/llm-morph)**|
|**2024-08-20**|**Towards Rehearsal-Free Multilingual ASR: A LoRA-based Case Study on Whisper**|Tianyi Xu et.al.|[2408.10680](http://arxiv.org/abs/2408.10680)|null|
|**2024-08-20**|**CoRA: Collaborative Information Perception by Large Language Model's Weights for Recommendation**|Yuting Liu et.al.|[2408.10645](http://arxiv.org/abs/2408.10645)|null|
|**2024-08-18**|**NoRA: Nested Low-Rank Adaptation for Efficient Fine-Tuning Large Models**|Cheng Lin et.al.|[2408.10280](http://arxiv.org/abs/2408.10280)|null|
|**2024-08-19**|**SMILE: Zero-Shot Sparse Mixture of Low-Rank Experts Construction From Pre-Trained Foundation Models**|Anke Tang et.al.|[2408.10174](http://arxiv.org/abs/2408.10174)|**[link](https://github.com/tanganke/fusion_bench)**|
|**2024-08-19**|**Customizing Language Models with Instance-wise LoRA for Sequential Recommendation**|Xiaoyu Kong et.al.|[2408.10159](http://arxiv.org/abs/2408.10159)|**[link](https://github.com/akalikong/ilora)**|
|**2024-08-19**|**TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition**|Tianwei Lin et.al.|[2408.09856](http://arxiv.org/abs/2408.09856)|**[link](https://github.com/lin-tianwei/teamlora)**|
|**2024-08-18**|**Infinite Scrolling, Finite Satisfaction: Exploring User Behavior and Satisfaction on Social Media in Bangladesh**|Sanzana Karim Lora et.al.|[2408.09601](http://arxiv.org/abs/2408.09601)|null|
|**2024-08-17**|**ConVerSum: A Contrastive Learning based Approach for Data-Scarce Solution of Cross-Lingual Summarization Beyond Direct Equivalents**|Sanzana Karim Lora et.al.|[2408.09273](http://arxiv.org/abs/2408.09273)|null|
|**2024-08-17**|**An Exploratory Study on Fine-Tuning Large Language Models for Secure Code Generation**|Junjie Li et.al.|[2408.09078](http://arxiv.org/abs/2408.09078)|**[link](https://github.com/SecureLLM/Secure_LLM)**|
|**2024-08-17**|**MoRA: LoRA Guided Multi-Modal Disease Diagnosis with Missing Modality**|Zhiyi Shi et.al.|[2408.09064](http://arxiv.org/abs/2408.09064)|null|
|**2024-08-16**|**AdaRank: Disagreement Based Module Rank Prediction for Low-rank Adaptation**|Yihe Dong et.al.|[2408.09015](http://arxiv.org/abs/2408.09015)|**[link](https://github.com/google-research/google-research)**|
|**2024-08-16**|**ML Study of MaliciousTransactions in Ethereum**|Natan Katz et.al.|[2408.08749](http://arxiv.org/abs/2408.08749)|null|
|**2024-08-16**|**RBLA: Rank-Based-LoRA-Aggregation for Fine-tuning Heterogeneous Models in FLaaS**|Shuaijun Chen et.al.|[2408.08699](http://arxiv.org/abs/2408.08699)|null|
|**2024-08-16**|**LLM-PCGC: Large Language Model-based Point Cloud Geometry Compression**|Yuqi Ye et.al.|[2408.08682](http://arxiv.org/abs/2408.08682)|null|
|**2024-08-16**|**Adaptive Layer Selection for Efficient Vision Transformer Fine-Tuning**|Alessio Devoto et.al.|[2408.08670](http://arxiv.org/abs/2408.08670)|null|
|**2024-08-16**|**A New Chinese Landscape Paintings Generation Model based on Stable Diffusion using DreamBooth**|Yujia Gu et.al.|[2408.08561](http://arxiv.org/abs/2408.08561)|null|
|**2024-08-15**|**Heavy Labels Out! Dataset Distillation with Label Space Lightening**|Ruonan Yu et.al.|[2408.08201](http://arxiv.org/abs/2408.08201)|null|
|**2024-08-15**|**When Video Coding Meets Multimodal Large Language Models: A Unified Paradigm for Video Coding**|Pingping Zhang et.al.|[2408.08093](http://arxiv.org/abs/2408.08093)|null|
|**2024-08-14**|**Domain-invariant Representation Learning via Segment Anything Model for Blood Cell Classification**|Yongcheng Li et.al.|[2408.07467](http://arxiv.org/abs/2408.07467)|**[link](https://github.com/anok3111/dorl)**|
|**2024-08-13**|**SeLoRA: Self-Expanding Low-Rank Adaptation of Latent Diffusion Model for Medical Image Synthesis**|Yuchen Mao et.al.|[2408.07196](http://arxiv.org/abs/2408.07196)|null|
|**2024-08-13**|**Imagen 3**|Imagen-Team-Google et.al.|[2408.07009](http://arxiv.org/abs/2408.07009)|null|
|**2024-08-13**|**New refinements of Narayana polynomials and Motzkin polynomials**|Janet J. W. Dong et.al.|[2408.06912](http://arxiv.org/abs/2408.06912)|null|
|**2024-08-13**|**LoRA $^2$ : Multi-Scale Low-Rank Approximations for Fine-Tuning Large Language Models**|Jia-Chen Zhang et.al.|[2408.06854](http://arxiv.org/abs/2408.06854)|null|
|**2024-08-13**|**DiffLoRA: Generating Personalized Low-Rank Adaptation Weights with Diffusion**|Yujia Wu et.al.|[2408.06740](http://arxiv.org/abs/2408.06740)|null|
|**2024-08-13**|**Towards Cross-Domain Single Blood Cell Image Classification via Large-Scale LoRA-based Segment Anything Model**|Yongcheng Li et.al.|[2408.06716](http://arxiv.org/abs/2408.06716)|**[link](https://github.com/anok3111/bc-sam)**|
|**2024-08-13**|**Harnessing Earnings Reports for Stock Predictions: A QLoRA-Enhanced LLM Approach**|Haowei Ni et.al.|[2408.06634](http://arxiv.org/abs/2408.06634)|null|
|**2024-08-13**|**Towards Robust and Cost-Efficient Knowledge Unlearning for Large Language Models**|Sungmin Cha et.al.|[2408.06621](http://arxiv.org/abs/2408.06621)|null|
|**2024-08-15**|**ControlNeXt: Powerful and Efficient Control for Image and Video Generation**|Bohao Peng et.al.|[2408.06070](http://arxiv.org/abs/2408.06070)|**[link](https://github.com/dvlab-research/controlnext)**|
|**2024-08-11**|**Hotfixing Large Language Models for Cod**|Zhou Yang et.al.|[2408.05727](http://arxiv.org/abs/2408.05727)|null|
|**2024-08-09**|**TaSL: Task Skill Localization and Consolidation for Language Model Continual Learning**|Yujie Feng et.al.|[2408.05200](http://arxiv.org/abs/2408.05200)|**[link](https://github.com/WoodScene/TaSL)**|
|**2024-08-09**|**LLaVA-VSD: Large Language-and-Vision Assistant for Visual Spatial Description**|Yizhang Jin et.al.|[2408.04957](http://arxiv.org/abs/2408.04957)|**[link](https://github.com/swordlidev/llava-vsd)**|
|**2024-08-09**|**Energy performance of LR-FHSS: analysis and evaluation**|Roger Sanchez-Vital et.al.|[2408.04908](http://arxiv.org/abs/2408.04908)|null|
|**2024-08-08**|**Bias-Aware Low-Rank Adaptation: Mitigating Catastrophic Inheritance of Large Language Models**|Yupeng Chang et.al.|[2408.04556](http://arxiv.org/abs/2408.04556)|**[link](https://github.com/cyp-jlu-ai/ba-lora)**|
|**2024-08-08**|**UNLEARN Efficient Removal of Knowledge in Large Language Models**|Tyler Lizzo et.al.|[2408.04140](http://arxiv.org/abs/2408.04140)|null|
|**2024-08-07**|**Image-to-LaTeX Converter for Mathematical Formulas and Text**|Daniil Gurgurov et.al.|[2408.04015](http://arxiv.org/abs/2408.04015)|**[link](https://github.com/d-gurgurov/im2latex)**|
|**2024-08-07**|**Speaker Adaptation for Quantised End-to-End ASR Models**|Qiuming Zhao et.al.|[2408.03979](http://arxiv.org/abs/2408.03979)|null|
|**2024-08-07**|**A Comparison of LLM Finetuning Methods & Evaluation Metrics with Travel Chatbot Use Case**|Sonia Meyer et.al.|[2408.03562](http://arxiv.org/abs/2408.03562)|null|
|**2024-08-11**|**Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation**|Jiachen Zhu et.al.|[2408.03533](http://arxiv.org/abs/2408.03533)|null|
|**2024-08-06**|**FastEdit: Fast Text-Guided Single-Image Editing via Semantic-Aware Diffusion Fine-Tuning**|Zhi Chen et.al.|[2408.03355](http://arxiv.org/abs/2408.03355)|null|
|**2024-08-06**|**SARA: Singular-Value Based Adaptive Low-Rank Adaption**|Jihao Gu et.al.|[2408.03290](http://arxiv.org/abs/2408.03290)|null|
|**2024-08-06**|**Leveraging Parameter Efficient Training Methods for Low Resource Text Classification: A Case Study in Marathi**|Pranita Deshmukh et.al.|[2408.03172](http://arxiv.org/abs/2408.03172)|null|
|**2024-08-06**|**L3iTC at the FinLLM Challenge Task: Quantization for Financial Text Classification & Summarization**|Elvys Linhares Pontes et.al.|[2408.03033](http://arxiv.org/abs/2408.03033)|null|
|**2024-08-06**|**Towards Smart Microfarming in an Urban Computing Continuum**|Marla Grunewald et.al.|[2408.02992](http://arxiv.org/abs/2408.02992)|null|
|**2024-08-05**|**StreamVoice+: Evolving into End-to-end Streaming Zero-shot Voice Conversion**|Zhichao Wang et.al.|[2408.02178](http://arxiv.org/abs/2408.02178)|null|
|**2024-08-04**|**SR-CIS: Self-Reflective Incremental System with Decoupled Memory and Reasoning**|Biqing Qi et.al.|[2408.01970](http://arxiv.org/abs/2408.01970)|null|
|**2024-08-03**|**Music2P: A Multi-Modal AI-Driven Tool for Simplifying Album Cover Design**|Joong Ho Choi et.al.|[2408.01651](http://arxiv.org/abs/2408.01651)|**[link](https://github.com/jc-78/music2p)**|
|**2024-08-02**|**MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts**|Lin Ning et.al.|[2408.01505](http://arxiv.org/abs/2408.01505)|null|
|**2024-08-02**|**Conditional LoRA Parameter Generation**|Xiaolong Jin et.al.|[2408.01415](http://arxiv.org/abs/2408.01415)|null|
|**2024-08-02**|**Pre-trained Language Models Improve the Few-shot Prompt Ability of Decision Transformer**|Yu Yang et.al.|[2408.01402](http://arxiv.org/abs/2408.01402)|null|
|**2024-08-02**|**Contribution-based Low-Rank Adaptation with Pre-training Model for Real Image Restoration**|Donwon Park et.al.|[2408.01099](http://arxiv.org/abs/2408.01099)|null|
|**2024-08-02**|**Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs**|Afia Anjum et.al.|[2408.01008](http://arxiv.org/abs/2408.01008)|null|
|**2024-08-02**|**PERSOMA: PERsonalized SOft ProMpt Adapter Architecture for Personalized Language Prompting**|Liam Hebert et.al.|[2408.00960](http://arxiv.org/abs/2408.00960)|null|
|**2024-08-01**|**Reclaiming Residual Knowledge: A Novel Paradigm to Low-Bit Quantization**|Róisín Luo et.al.|[2408.00923](http://arxiv.org/abs/2408.00923)|null|
|**2024-07-31**|**Ge-based Clinopyroxene series: first principles and experimental local probe study**|Ricardo P. Moreira et.al.|[2407.21749](http://arxiv.org/abs/2407.21749)|null|
|**2024-07-31**|**A Federated Learning-Friendly Approach for Parameter-Efficient Fine-Tuning of SAM in 3D Segmentation**|Mothilal Asokan et.al.|[2407.21739](http://arxiv.org/abs/2407.21739)|null|
|**2024-07-31**|**Zero-Shot Cross-Domain Dialogue State Tracking via Dual Low-Rank Adaptation**|Xiang Luo et.al.|[2407.21633](http://arxiv.org/abs/2407.21633)|**[link](https://github.com/suntea233/duallora)**|
|**2024-07-30**|**CELLM: An Efficient Communication in Large Language Models Training for Federated Learning**|Raja Vavekanand et.al.|[2407.20557](http://arxiv.org/abs/2407.20557)|null|
|**2024-07-29**|**Generative Diffusion Model Bootstraps Zero-shot Classification of Fetal Ultrasound Images In Underrepresented African Populations**|Fangyijie Wang et.al.|[2407.20072](http://arxiv.org/abs/2407.20072)|**[link](https://github.com/13204942/fu-lora)**|
|**2024-07-28**|**Memory-efficient Training of LLMs with Larger Mini-batches**|Dang Nguyen et.al.|[2407.19580](http://arxiv.org/abs/2407.19580)|null|
|**2024-07-27**|**Parameter-Efficient Fine-Tuning via Circular Convolution**|Aochuan Chen et.al.|[2407.19342](http://arxiv.org/abs/2407.19342)|null|
|**2024-07-27**|**The Impact of LoRA Adapters for LLMs on Clinical NLP Classification Under Data Limitations**|Thanh-Dung Le et.al.|[2407.19299](http://arxiv.org/abs/2407.19299)|null|
|**2024-07-26**|**VIMs: Virtual Immunohistochemistry Multiplex staining via Text-to-Stain Diffusion Trained on Uniplex Stains**|Shikha Dubey et.al.|[2407.19113](http://arxiv.org/abs/2407.19113)|null|
|**2024-07-25**|**Stay Tuned: An Empirical Study of the Impact of Hyperparameters on LLM Tuning in Real-World Applications**|Alon Halfon et.al.|[2407.18990](http://arxiv.org/abs/2407.18990)|null|
|**2024-07-25**|**LoRA-Pro: Are Low-Rank Adapters Properly Optimized?**|Zhengbo Wang et.al.|[2407.18242](http://arxiv.org/abs/2407.18242)|**[link](https://github.com/mrflogs/LoRA-Pro)**|
|**2024-07-25**|**DINOv2 Rocks Geological Image Analysis: Classification, Segmentation, and Interpretability**|Florent Brondolo et.al.|[2407.18100](http://arxiv.org/abs/2407.18100)|**[link](https://github.com/flofive/dinov2-x-geosciences)**|
|**2024-07-24**|**Channel-Aware Low-Rank Adaptation in Time Series Forecasting**|Tong Nie et.al.|[2407.17246](http://arxiv.org/abs/2407.17246)|**[link](https://github.com/tongnie/c-lora)**|
|**2024-07-24**|**Accurate and Efficient Fine-Tuning of Quantized Large Language Models Through Optimal Balance**|Ao Shen et.al.|[2407.17029](http://arxiv.org/abs/2407.17029)|**[link](https://github.com/xiaocaigou/qbaraqahira)**|
|**2024-07-22**|**Rapid Switching and Multi-Adapter Fusion via Sparse High Rank Adapters**|Kartikeya Bhardwaj et.al.|[2407.16712](http://arxiv.org/abs/2407.16712)|null|
|**2024-07-23**|**DreamVTON: Customizing 3D Virtual Try-on with Personalized Diffusion Models**|Zhenyu Xie et.al.|[2407.16511](http://arxiv.org/abs/2407.16511)|null|
|**2024-07-23**|**Harmonizing Visual Text Comprehension and Generation**|Zhen Zhao et.al.|[2407.16364](http://arxiv.org/abs/2407.16364)|**[link](https://github.com/bytedance/textharmony)**|
|**2024-07-23**|**FoRA: Low-Rank Adaptation Model beyond Multimodal Siamese Network**|Weiying Xie et.al.|[2407.16129](http://arxiv.org/abs/2407.16129)|**[link](https://github.com/zyszxhy/fora)**|
|**2024-07-22**|**Test-Time Low Rank Adaptation via Confidence Maximization for Zero-Shot Generalization of Vision-Language Models**|Raza Imam et.al.|[2407.15913](http://arxiv.org/abs/2407.15913)|**[link](https://github.com/razaimam45/ttl-test-time-low-rank-adaptation)**|
|**2024-07-22**|**Zero-Shot Embeddings Inform Learning and Forgetting with Vision-Language Encoders**|Laura Niss et.al.|[2407.15731](http://arxiv.org/abs/2407.15731)|null|
|**2024-07-22**|**LLaST: Improved End-to-end Speech Translation System Leveraged by Large Language Models**|Xi Chen et.al.|[2407.15415](http://arxiv.org/abs/2407.15415)|**[link](https://github.com/openaudiolab/llast)**|
|**2024-07-21**|**Learn to Preserve and Diversify: Parameter-Efficient Group with Orthogonal Regularization for Domain Generalization**|Jiajun Hu et.al.|[2407.15085](http://arxiv.org/abs/2407.15085)|null|
|**2024-07-21**|**MedSAGa: Few-shot Memory Efficient Medical Image Segmentation using Gradient Low-Rank Projection in SAM**|Navyansh Mahla et.al.|[2407.15042](http://arxiv.org/abs/2407.15042)|null|

<p align=right>(<a href=#updated-on-20241231>back to top</a>)</p>

## Model Compression

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-12-30**|**Improving Acoustic Scene Classification in Low-Resource Conditions**|Zhi Chen et.al.|[2412.20722](http://arxiv.org/abs/2412.20722)|null|
|**2024-12-28**|**Injecting Explainability and Lightweight Design into Weakly Supervised Video Anomaly Detection Systems**|Wen-Dong Jiang et.al.|[2412.20201](http://arxiv.org/abs/2412.20201)|null|
|**2024-12-28**|**SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection**|Phi Vu Tran et.al.|[2412.20047](http://arxiv.org/abs/2412.20047)|null|
|**2024-12-28**|**Invariant debiasing learning for recommendation via biased imputation**|Ting Bai et.al.|[2412.20036](http://arxiv.org/abs/2412.20036)|**[link](https://github.com/bai-lab/kd-debias)**|
|**2024-12-28**|**Learning Adaptive and View-Invariant Vision Transformer with Multi-Teacher Knowledge Distillation for Real-Time UAV Tracking**|You Wu et.al.|[2412.20002](http://arxiv.org/abs/2412.20002)|null|
|**2024-12-27**|**Asymmetrical Reciprocity-based Federated Learning for Resolving Disparities in Medical Diagnosis**|Jiaqi Wang et.al.|[2412.19654](http://arxiv.org/abs/2412.19654)|null|
|**2024-12-27**|**Feature Alignment-Based Knowledge Distillation for Efficient Compression of Large Language Models**|Shuo Wang et.al.|[2412.19449](http://arxiv.org/abs/2412.19449)|null|
|**2024-12-26**|**SpectralKD: Understanding and Optimizing Vision Transformer Distillation through Spectral Analysis**|Huiyuan Tian et.al.|[2412.19055](http://arxiv.org/abs/2412.19055)|null|
|**2024-12-25**|**Optimization and Scalability of Collaborative Filtering Algorithms in Large Language Models**|Haowei Yang et.al.|[2412.18715](http://arxiv.org/abs/2412.18715)|null|
|**2024-12-23**|**Edge-AI for Agriculture: Lightweight Vision Models for Disease Detection in Resource-Limited Settings**|Harsh Joshi et.al.|[2412.18635](http://arxiv.org/abs/2412.18635)|null|
|**2024-12-24**|**HTR-JAND: Handwritten Text Recognition with Joint Attention Network and Knowledge Distillation**|Mohammed Hamdan et.al.|[2412.18524](http://arxiv.org/abs/2412.18524)|null|
|**2024-12-24**|**Understanding Artificial Neural Network's Behavior from Neuron Activation Perspective**|Yizhou Zhang et.al.|[2412.18073](http://arxiv.org/abs/2412.18073)|null|
|**2024-12-23**|**CoSurfGS:Collaborative 3D Surface Gaussian Splatting with Distributed Learning for Large Scene Reconstruction**|Yuanyuan Gao et.al.|[2412.17612](http://arxiv.org/abs/2412.17612)|null|
|**2024-12-23**|**GQSA: Group Quantization and Sparsity for Accelerating Large Language Model Inference**|Chao Zeng et.al.|[2412.17560](http://arxiv.org/abs/2412.17560)|null|
|**2024-12-24**|**Singular Value Scaling: Efficient Generative Model Compression via Pruned Weights Refinement**|Hyeonjin Kim et.al.|[2412.17387](http://arxiv.org/abs/2412.17387)|**[link](https://github.com/lait-cvlab/singular-value-scaling)**|
|**2024-12-23**|**Better Knowledge Enhancement for Privacy-Preserving Cross-Project Defect Prediction**|Yuying Wang et.al.|[2412.17317](http://arxiv.org/abs/2412.17317)|null|
|**2024-12-23**|**LMD-PGN: Cross-Modal Knowledge Distillation from First-Person-View Images to Third-Person-View BEV Maps for Universal Point Goal Navigation**|Riku Uemura et.al.|[2412.17282](http://arxiv.org/abs/2412.17282)|null|
|**2024-12-22**|**Lightweight Design and Optimization methods for DCNNs: Progress and Futures**|Hanhua Long et.al.|[2412.16886](http://arxiv.org/abs/2412.16886)|null|
|**2024-12-21**|**Large Language Models Compression via Low-Rank Feature Distillation**|Yaya Sy et.al.|[2412.16719](http://arxiv.org/abs/2412.16719)|null|
|**2024-12-21**|**CyberSentinel: Efficient Anomaly Detection in Programmable Switch using Knowledge Distillation**|Sankalp Mittal et.al.|[2412.16693](http://arxiv.org/abs/2412.16693)|null|
|**2024-12-21**|**Semantics Prompting Data-Free Quantization for Low-Bit Vision Transformers**|Yunshan Zhong et.al.|[2412.16553](http://arxiv.org/abs/2412.16553)|null|
|**2024-12-21**|**STKDRec: Spatial-Temporal Knowledge Distillation for Takeaway Recommendation**|Shuyuan Zhao et.al.|[2412.16502](http://arxiv.org/abs/2412.16502)|null|
|**2024-12-20**|**BabyHGRN: Exploring RNNs for Sample-Efficient Training of Language Models**|Patrick Haller et.al.|[2412.15978](http://arxiv.org/abs/2412.15978)|null|
|**2024-12-20**|**A New Method to Capturing Compositional Knowledge in Linguistic Space**|Jiahe Wan et.al.|[2412.15632](http://arxiv.org/abs/2412.15632)|null|
|**2024-12-19**|**Uncertainty-Guided Cross Attention Ensemble Mean Teacher for Semi-supervised Medical Image Segmentation**|Meghana Karri et.al.|[2412.15380](http://arxiv.org/abs/2412.15380)|null|
|**2024-12-19**|**Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models**|Reza Shirkavand et.al.|[2412.15341](http://arxiv.org/abs/2412.15341)|null|
|**2024-12-19**|**Self-Evolution Knowledge Distillation for LLM-based Machine Translation**|Yuncheng Song et.al.|[2412.15303](http://arxiv.org/abs/2412.15303)|null|
|**2024-12-19**|**Adaptive Pruning for Large Language Models with Structural Importance Awareness**|Haotian Zheng et.al.|[2412.15127](http://arxiv.org/abs/2412.15127)|null|
|**2024-12-19**|**SCKD: Semi-Supervised Cross-Modality Knowledge Distillation for 4D Radar Object Detection**|Ruoyu Xu et.al.|[2412.14571](http://arxiv.org/abs/2412.14571)|null|
|**2024-12-19**|**Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models**|Xiao Cui et.al.|[2412.14528](http://arxiv.org/abs/2412.14528)|null|
|**2024-12-19**|**Knowledge Distillation in RNN-Attention Models for Early Prediction of Student Performance**|Sukrit Leelaluk et.al.|[2412.14526](http://arxiv.org/abs/2412.14526)|**[link](https://github.com/limu-research/2025-sac-rnn-attention-kd)**|
|**2024-12-18**|**A Survey on Inference Optimization Techniques for Mixture of Experts Models**|Jiacheng Liu et.al.|[2412.14219](http://arxiv.org/abs/2412.14219)|**[link](https://github.com/moe-inf/awesome-moe-inference)**|
|**2024-12-18**|**Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective**|Zhiyuan Zeng et.al.|[2412.14135](http://arxiv.org/abs/2412.14135)|null|
|**2024-12-18**|**On Explaining Knowledge Distillation: Measuring and Visualising the Knowledge Transfer Process**|Gereziher Adhane et.al.|[2412.13943](http://arxiv.org/abs/2412.13943)|null|
|**2024-12-18**|**Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN**|Pengxiang Li et.al.|[2412.13795](http://arxiv.org/abs/2412.13795)|**[link](https://github.com/pixeli99/mixln)**|
|**2024-12-18**|**Learnable Prompting SAM-induced Knowledge Distillation for Semi-supervised Medical Image Segmentation**|Kaiwen Huang et.al.|[2412.13742](http://arxiv.org/abs/2412.13742)|**[link](https://github.com/taozh2017/knowsam)**|
|**2024-12-18**|**On the Compression of Language Models for Code: An Empirical Study on CodeBERT**|Giordano d'Aloisio et.al.|[2412.13737](http://arxiv.org/abs/2412.13737)|null|
|**2024-12-18**|**Hybrid Data-Free Knowledge Distillation**|Jialiang Tang et.al.|[2412.13525](http://arxiv.org/abs/2412.13525)|**[link](https://github.com/tangjialiang97/hidfd)**|
|**2024-12-18**|**Deploying Foundation Model Powered Agent Services: A Survey**|Wenchao Xu et.al.|[2412.13437](http://arxiv.org/abs/2412.13437)|null|
|**2024-12-17**|**In-Context Learning Distillation for Efficient Few-Shot Fine-Tuning**|Yifei Duan et.al.|[2412.13243](http://arxiv.org/abs/2412.13243)|null|
|**2024-12-17**|**Modality-Inconsistent Continual Learning of Multimodal Large Language Models**|Weiguo Pian et.al.|[2412.13050](http://arxiv.org/abs/2412.13050)|null|
|**2024-12-17**|**Efficient Speech Command Recognition Leveraging Spiking Neural Network and Curriculum Learning-based Knowledge Distillation**|Jiaqi Wang et.al.|[2412.12858](http://arxiv.org/abs/2412.12858)|null|
|**2024-12-17**|**RemoteTrimmer: Adaptive Structural Pruning for Remote Sensing Image Classification**|Guanwenjie Zou et.al.|[2412.12603](http://arxiv.org/abs/2412.12603)|**[link](https://github.com/1e12Leon/RemoteTrimmer)**|
|**2024-12-17**|**PromptDet: A Lightweight 3D Object Detection Framework with LiDAR Prompts**|Kun Guo et.al.|[2412.12460](http://arxiv.org/abs/2412.12460)|**[link](https://github.com/lihuashengmax/PromptDet)**|
|**2024-12-16**|**Neural Collapse Inspired Knowledge Distillation**|Shuoxi Zhang et.al.|[2412.11788](http://arxiv.org/abs/2412.11788)|null|
|**2024-12-16**|**Relation-Guided Adversarial Learning for Data-free Knowledge Transfer**|Yingping Liang et.al.|[2412.11380](http://arxiv.org/abs/2412.11380)|**[link](https://github.com/sharpiless/rgal)**|
|**2024-12-16**|**BiM-VFI: directional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions**|Wonyong Seo et.al.|[2412.11365](http://arxiv.org/abs/2412.11365)|null|
|**2024-12-15**|**Wearable Accelerometer Foundation Models for Health via Knowledge Distillation**|Salar Abbaspourazad et.al.|[2412.11276](http://arxiv.org/abs/2412.11276)|null|
|**2024-12-15**|**TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs**|Lanxiang Hu et.al.|[2412.11242](http://arxiv.org/abs/2412.11242)|null|
|**2024-12-15**|**ProFe: Communication-Efficient Decentralized Federated Learning via Distillation and Prototypes**|Pedro Miguel Sánchez Sánchez et.al.|[2412.11207](http://arxiv.org/abs/2412.11207)|null|
|**2024-12-15**|**Leveraging Large Language Models for Active Merchant Non-player Characters**|Byungjun Kim et.al.|[2412.11189](http://arxiv.org/abs/2412.11189)|null|
|**2024-12-15**|**Knowledge Migration Framework for Smart Contract Vulnerability Detection**|Luqi Wang et.al.|[2412.11175](http://arxiv.org/abs/2412.11175)|null|
|**2024-12-15**|**Redefining Normal: A Novel Object-Level Approach for Multi-Object Novelty Detection**|Mohammadreza Salehi et.al.|[2412.11148](http://arxiv.org/abs/2412.11148)|**[link](https://github.com/smsd75/redefining_normal_accv24)**|
|**2024-12-17**|**On Distilling the Displacement Knowledge for Few-Shot Class-Incremental Learning**|Pengfei Fang et.al.|[2412.11017](http://arxiv.org/abs/2412.11017)|null|
|**2024-12-13**|**Can Students Beyond The Teacher? Distilling Knowledge from Teacher's Bias**|Jianhua Zhang et.al.|[2412.09874](http://arxiv.org/abs/2412.09874)|null|
|**2024-12-13**|**ScaleOT: Privacy-utility-scalable Offsite-tuning with Dynamic LayerReplace and Selective Rank Compression**|Kai Yao et.al.|[2412.09812](http://arxiv.org/abs/2412.09812)|null|
|**2024-12-13**|**LLM Distillation for Efficient Few-Shot Multiple Choice Question Answering**|Patrick Sutanto et.al.|[2412.09807](http://arxiv.org/abs/2412.09807)|null|
|**2024-12-12**|**SnapGen: Taming High-Resolution Text-to-Image Models for Mobile Devices with Efficient Architectures and Training**|Dongting Hu et.al.|[2412.09619](http://arxiv.org/abs/2412.09619)|null|
|**2024-12-12**|**A Theoretical Analysis of Soft-Label vs Hard-Label Training in Neural Networks**|Saptarshi Mandal et.al.|[2412.09579](http://arxiv.org/abs/2412.09579)|null|
|**2024-12-12**|**All You Need in Knowledge Distillation Is a Tailored Coordinate System**|Junjie Zhou et.al.|[2412.09388](http://arxiv.org/abs/2412.09388)|null|
|**2024-12-12**|**Optimising TinyML with Quantization and Distillation of Transformer and Mamba Models for Indoor Localisation on Edge Devices**|Thanaphon Suwannaphong et.al.|[2412.09289](http://arxiv.org/abs/2412.09289)|null|
|**2024-12-15**|**DASK: Distribution Rehearsing via Adaptive Style Kernel Learning for Exemplar-Free Lifelong Person Re-Identification**|Kunlun Xu et.al.|[2412.09224](http://arxiv.org/abs/2412.09224)|**[link](https://github.com/zhoujiahuan1991/aaai2025-dask)**|
|**2024-12-12**|**Multimodal Industrial Anomaly Detection by Crossmodal Reverse Distillation**|Xinyue Liu et.al.|[2412.08949](http://arxiv.org/abs/2412.08949)|**[link](https://github.com/hito2448/CRD)**|
|**2024-12-12**|**Dynamic Contrastive Knowledge Distillation for Efficient Image Restoration**|Yunshuai Zhou et.al.|[2412.08939](http://arxiv.org/abs/2412.08939)|null|
|**2024-12-11**|**Efficient Gravitational Wave Parameter Estimation via Knowledge Distillation: A ResNet1D-IAF Approach**|Xihua Zhu et.al.|[2412.08672](http://arxiv.org/abs/2412.08672)|null|
|**2024-12-11**|**Wasserstein Distance Rivals Kullback-Leibler Divergence for Knowledge Distillation**|Jiaming Lv et.al.|[2412.08139](http://arxiv.org/abs/2412.08139)|null|
|**2024-12-11**|**DAKD: Data Augmentation and Knowledge Distillation using Diffusion Models for SAR Oil Spill Segmentation**|Jaeho Moon et.al.|[2412.08116](http://arxiv.org/abs/2412.08116)|null|
|**2024-12-10**|**Low-Rank Correction for Quantized LLMs**|Meyer Scetbon et.al.|[2412.07902](http://arxiv.org/abs/2412.07902)|null|
|**2024-12-10**|**Unlocking the Potential of Reverse Distillation for Anomaly Detection**|Xinyue Liu et.al.|[2412.07579](http://arxiv.org/abs/2412.07579)|**[link](https://github.com/hito2448/urd)**|
|**2024-12-10**|**TT-MPD: Test Time Model Pruning and Distillation**|Haihang Wu et.al.|[2412.07114](http://arxiv.org/abs/2412.07114)|null|
|**2024-12-09**|**FM2DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering**|Amirhossein Abaskohi et.al.|[2412.07030](http://arxiv.org/abs/2412.07030)|**[link](https://github.com/servicenow/fm2ds)**|
|**2024-12-09**|**VQ4ALL: Efficient Neural Network Representation via a Universal Codebook**|Juncan Deng et.al.|[2412.06875](http://arxiv.org/abs/2412.06875)|null|
|**2024-12-09**|**Compression for Better: A General and Stable Lossless Compression Framework**|Boyang Zhang et.al.|[2412.06868](http://arxiv.org/abs/2412.06868)|null|
|**2024-12-09**|**Lossless Model Compression via Joint Low-Rank Factorization Optimization**|Boyang Zhang et.al.|[2412.06867](http://arxiv.org/abs/2412.06867)|null|
|**2024-12-08**|**GL-Fusion: Rethinking the Combination of Graph Neural Network and Large Language model**|Haotong Yang et.al.|[2412.06849](http://arxiv.org/abs/2412.06849)|null|
|**2024-12-10**|**Federated Split Learning with Model Pruning and Gradient Quantization in Wireless Networks**|Junhe Zhang et.al.|[2412.06414](http://arxiv.org/abs/2412.06414)|null|
|**2024-12-09**|**U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening**|Sungpyo Kim et.al.|[2412.06243](http://arxiv.org/abs/2412.06243)|null|
|**2024-12-08**|**Enhancing Content Representation for AR Image Quality Assessment Using Knowledge Distillation**|Aymen Sekhri et.al.|[2412.06003](http://arxiv.org/abs/2412.06003)|null|
|**2024-12-07**|**Neighborhood Commonality-aware Evolution Network for Continuous Generalized Category Discovery**|Ye Wang et.al.|[2412.05573](http://arxiv.org/abs/2412.05573)|null|
|**2024-12-07**|**Trimming Down Large Spiking Vision Transformers via Heterogeneous Quantization Search**|Boxun Xu et.al.|[2412.05505](http://arxiv.org/abs/2412.05505)|null|
|**2024-12-06**|**BEExformer: A Fast Inferencing Transformer Architecture via Binarization with Multiple Early Exits**|Wazib Ansar et.al.|[2412.05225](http://arxiv.org/abs/2412.05225)|null|
|**2024-12-06**|**One-shot Federated Learning via Synthetic Distiller-Distillate Communication**|Junyuan Zhang et.al.|[2412.05186](http://arxiv.org/abs/2412.05186)|**[link](https://github.com/carkham/fedsd2c)**|
|**2024-12-06**|**CCS: Continuous Learning for Customized Incremental Wireless Sensing Services**|Qunhang Fu et.al.|[2412.04821](http://arxiv.org/abs/2412.04821)|null|
|**2024-12-05**|**Diffusion-Augmented Coreset Expansion for Scalable Dataset Distillation**|Ali Abbasi et.al.|[2412.04668](http://arxiv.org/abs/2412.04668)|null|
|**2024-12-05**|**FedDW: Distilling Weights through Consistency Optimization in Heterogeneous Federated Learning**|Jiayu Liu et.al.|[2412.04521](http://arxiv.org/abs/2412.04521)|**[link](https://github.com/liuvvvvv1/feddw)**|
|**2024-12-05**|**Expanding Deep Learning-based Sensing Systems with Multi-Source Knowledge Transfer**|Gaole Dai et.al.|[2412.04060](http://arxiv.org/abs/2412.04060)|null|
|**2024-12-04**|**Designing DNNs for a trade-off between robustness and processing performance in embedded devices**|Jon Gutiérrez-Zaballa et.al.|[2412.03682](http://arxiv.org/abs/2412.03682)|null|
|**2024-12-04**|**Evaluating Single Event Upsets in Deep Neural Networks for Semantic Segmentation: an embedded system perspective**|Jon Gutiérrez-Zaballa et.al.|[2412.03630](http://arxiv.org/abs/2412.03630)|**[link](https://github.com/jonguti13/parameterprotection)**|
|**2024-12-03**|**CPTQuant -- A Novel Mixed Precision Post-Training Quantization Techniques for Large Language Models**|Amitash Nanda et.al.|[2412.03599](http://arxiv.org/abs/2412.03599)|null|
|**2024-12-07**|**Enhancing CLIP Conceptual Embedding through Knowledge Distillation**|Kuei-Chun Kao et.al.|[2412.03513](http://arxiv.org/abs/2412.03513)|null|
|**2024-12-04**|**Distillation of Diffusion Features for Semantic Correspondence**|Frank Fundel et.al.|[2412.03512](http://arxiv.org/abs/2412.03512)|null|
|**2024-12-03**|**Efficient Model Compression Techniques with FishLeg**|Jamie McGowan et.al.|[2412.02328](http://arxiv.org/abs/2412.02328)|null|
|**2024-12-02**|**Mutli-View 3D Reconstruction using Knowledge Distillation**|Aditya Dutt et.al.|[2412.02039](http://arxiv.org/abs/2412.02039)|**[link](https://github.com/ishikaalunawat/231aproj)**|
|**2024-12-02**|**Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Model**|Qianhan Feng et.al.|[2412.01282](http://arxiv.org/abs/2412.01282)|**[link](https://github.com/fqhank/align-kd)**|
|**2024-12-02**|**Reducing Inference Energy Consumption Using Dual Complementary CNNs**|Michail Kinnas et.al.|[2412.01039](http://arxiv.org/abs/2412.01039)|**[link](https://github.com/michaelkinnas/Reducing-Inference-Energy-Consumption-Using-Two-Complementary-CNNs)**|
|**2024-12-01**|**QABISAR: Query-Article Bipartite Interactions for Statutory Article Retrieval**|T. Y. S. S. Santosh et.al.|[2412.00934](http://arxiv.org/abs/2412.00934)|null|
|**2024-12-01**|**Local vs. Global: Local Land-Use and Land-Cover Models Deliver Higher Quality Maps**|Girmaw Abebe Tadesse et.al.|[2412.00777](http://arxiv.org/abs/2412.00777)|null|
|**2024-11-30**|**Continuous Concepts Removal in Text-to-image Diffusion Models**|Tingxu Han et.al.|[2412.00580](http://arxiv.org/abs/2412.00580)|null|
|**2024-11-30**|**Pruned Convolutional Attention Network Based Wideband Spectrum Sensing with Sub-Nyquist Sampling**|Peihao Dong et.al.|[2412.00562](http://arxiv.org/abs/2412.00562)|**[link](https://github.com/AI4CogComm/PCA-WSSNet)**|
|**2024-11-30**|**Toward Fair Graph Neural Networks Via Dual-Teacher Knowledge Distillation**|Chengyu Li et.al.|[2412.00382](http://arxiv.org/abs/2412.00382)|null|
|**2024-11-29**|**Reverse Thinking Makes LLMs Stronger Reasoners**|Justin Chih-Yao Chen et.al.|[2411.19865](http://arxiv.org/abs/2411.19865)|null|
|**2024-11-28**|**Pre-Training Graph Contrastive Masked Autoencoders are Strong Distillers for EEG**|Xinxu Wei et.al.|[2411.19230](http://arxiv.org/abs/2411.19230)|null|
|**2024-12-03**|**Puzzle: Distillation-Based NAS for Inference-Optimized LLMs**|Akhiad Bercovich et.al.|[2411.19146](http://arxiv.org/abs/2411.19146)|null|
|**2024-11-28**|**Headache to Overstock? Promoting Long-tail Items through Debiased Product Bundling**|Shuo Xu et.al.|[2411.19107](http://arxiv.org/abs/2411.19107)|null|
|**2024-11-28**|**Zero-shot Slot Filling in the Age of LLMs for Dialogue Systems**|Mansi Rana et.al.|[2411.18980](http://arxiv.org/abs/2411.18980)|null|
|**2024-11-27**|**Active Data Curation Effectively Distills Large-Scale Multimodal Models**|Vishaal Udandarao et.al.|[2411.18674](http://arxiv.org/abs/2411.18674)|null|
|**2024-11-27**|**Individual Content and Motion Dynamics Preserved Pruning for Video Diffusion Models**|Yiming Wu et.al.|[2411.18375](http://arxiv.org/abs/2411.18375)|null|
|**2024-11-27**|**Vision Mamba Distillation for Low-resolution Fine-grained Image Classification**|Yao Chen et.al.|[2411.17980](http://arxiv.org/abs/2411.17980)|**[link](https://github.com/boa2004plaust/vimd)**|
|**2024-11-27**|**Improved implicit diffusion model with knowledge distillation to estimate the spatial distribution density of carbon stock in remote sensing imagery**|Zhenyu Yu et.al.|[2411.17973](http://arxiv.org/abs/2411.17973)|null|
|**2024-11-26**|**Attamba: Attending To Multi-Token States**|Yash Akhauri et.al.|[2411.17685](http://arxiv.org/abs/2411.17685)|**[link](https://github.com/abdelfattah-lab/attamba)**|
|**2024-11-26**|**Large-Scale Data-Free Knowledge Distillation for ImageNet via Multi-Resolution Data Generation**|Minh-Tuan Tran et.al.|[2411.17046](http://arxiv.org/abs/2411.17046)|null|
|**2024-11-26**|**Words Matter: Leveraging Individual Text Embeddings for Code Generation in CLIP Test-Time Adaptation**|Shambhavi Mishra et.al.|[2411.17002](http://arxiv.org/abs/2411.17002)|**[link](https://github.com/ShambhaviCodes/CLIPOT)**|
|**2024-11-25**|**Dynamic Self-Distillation via Previous Mini-batches for Fine-tuning Small Language Models**|Yao Fu et.al.|[2411.16991](http://arxiv.org/abs/2411.16991)|null|
|**2024-11-25**|**Leveraging Foundation Models To learn the shape of semi-fluid deformable objects**|Omar El Assal et.al.|[2411.16802](http://arxiv.org/abs/2411.16802)|null|
|**2024-11-25**|**O1 Replication Journey -- Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?**|Zhen Huang et.al.|[2411.16489](http://arxiv.org/abs/2411.16489)|**[link](https://github.com/gair-nlp/o1-journey)**|
|**2024-11-25**|**When Babies Teach Babies: Can student knowledge sharing outperform Teacher-Guided Distillation on small datasets?**|Srikrishna Iyer et.al.|[2411.16487](http://arxiv.org/abs/2411.16487)|**[link](https://github.com/ai-da-stc/generative-ai-research-babylm)**|
|**2024-11-25**|**Learn from Foundation Model: Fruit Detection Model without Manual Annotation**|Yanan Wang et.al.|[2411.16196](http://arxiv.org/abs/2411.16196)|**[link](https://github.com/agroboticsresearch/sdm-d)**|
|**2024-11-25**|**Beyond Task Vectors: Selective Task Arithmetic Based on Importance Metrics**|Tian Bowen et.al.|[2411.16139](http://arxiv.org/abs/2411.16139)|null|
|**2024-11-25**|**Ensemble Learning via Knowledge Transfer for CTR Prediction**|Honghao Li et.al.|[2411.16122](http://arxiv.org/abs/2411.16122)|**[link](https://github.com/salmon1802/ektf)**|
|**2024-11-23**|**Botfip-LLM: An Enhanced Multimodal Scientific Computing Framework Leveraging Knowledge Distillation from Large Language Models**|Tianhao Chen et.al.|[2411.15525](http://arxiv.org/abs/2411.15525)|null|
|**2024-11-23**|**Efficient Ternary Weight Embedding Model: Bridging Scalability and Performance**|Jiayi Chen et.al.|[2411.15438](http://arxiv.org/abs/2411.15438)|**[link](https://github.com/dataparameters/Ternary-Embedding-Models)**|
|**2024-11-23**|**Partial Knowledge Distillation for Alleviating the Inherent Inter-Class Discrepancy in Federated Learning**|Xiaoyu Gan et.al.|[2411.15403](http://arxiv.org/abs/2411.15403)|null|
|**2024-11-22**|**Efficient Pruning of Text-to-Image Models: Insights from Pruning Stable Diffusion**|Samarth N Ramesh et.al.|[2411.15113](http://arxiv.org/abs/2411.15113)|null|
|**2024-11-22**|**RankByGene: Gene-Guided Histopathology Representation Learning Through Cross-Modal Ranking Consistency**|Wentao Huang et.al.|[2411.15076](http://arxiv.org/abs/2411.15076)|null|
|**2024-11-22**|**Adaptive Group Robust Ensemble Knowledge Distillation**|Patrik Kenfack et.al.|[2411.14984](http://arxiv.org/abs/2411.14984)|null|
|**2024-11-25**|**Information Extraction from Heterogeneous Documents without Ground Truth Labels using Synthetic Label Generation and Knowledge Distillation**|Aniket Bhattacharyya et.al.|[2411.14957](http://arxiv.org/abs/2411.14957)|null|
|**2024-11-22**|**Simplifying CLIP: Unleashing the Power of Large-Scale Models on Consumer-level Computers**|Hongbo Liu et.al.|[2411.14789](http://arxiv.org/abs/2411.14789)|null|
|**2024-11-22**|**Improving Mathematical Reasoning Capabilities of Small Language Models via Feedback-Driven Distillation**|Xunyu Zhu et.al.|[2411.14698](http://arxiv.org/abs/2411.14698)|null|
|**2024-11-21**|**TaQ-DiT: Time-aware Quantization for Diffusion Transformers**|Xinyan Liu et.al.|[2411.14172](http://arxiv.org/abs/2411.14172)|null|
|**2024-11-21**|**DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization**|Hexuan Deng et.al.|[2411.14055](http://arxiv.org/abs/2411.14055)|**[link](https://github.com/hexuandeng/drpruning)**|
|**2024-11-21**|**Teaching MLPs to Master Heterogeneous Graph-Structured Knowledge for Efficient and Accurate Inference**|Yunhui Liu et.al.|[2411.14035](http://arxiv.org/abs/2411.14035)|**[link](https://github.com/cloudy1225/hg2m)**|
|**2024-11-21**|**CLFace: A Scalable and Resource-Efficient Continual Learning Framework for Lifelong Face Recognition**|Md Mahedi Hasan et.al.|[2411.13886](http://arxiv.org/abs/2411.13886)|null|
|**2024-11-20**|**RTSR: A Real-Time Super-Resolution Model for AV1 Compressed Content**|Yuxuan Jiang et.al.|[2411.13362](http://arxiv.org/abs/2411.13362)|null|
|**2024-11-20**|**FASTNav: Fine-tuned Adaptive Small-language-models Trained for Multi-point Robot Navigation**|Yuxuan Chen et.al.|[2411.13262](http://arxiv.org/abs/2411.13262)|null|
|**2024-11-20**|**Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning**|Gang Zhao et.al.|[2411.13045](http://arxiv.org/abs/2411.13045)|null|
|**2024-11-19**|**Puppet-CNN: Input-Adaptive Convolutional Neural Networks with Model Compression using Ordinary Differential Equation**|Yucheng Xing et.al.|[2411.12876](http://arxiv.org/abs/2411.12876)|null|
|**2024-11-19**|**Reward Modeling with Ordinal Feedback: Wisdom of the Crowd**|Shang Liu et.al.|[2411.12843](http://arxiv.org/abs/2411.12843)|null|
|**2024-11-19**|**What Makes a Good Dataset for Knowledge Distillation?**|Logan Frank et.al.|[2411.12817](http://arxiv.org/abs/2411.12817)|null|
|**2024-11-19**|**FGP: Feature-Gradient-Prune for Efficient Convolutional Layer Pruning**|Qingsong Lv et.al.|[2411.12781](http://arxiv.org/abs/2411.12781)|**[link](https://github.com/fgp-code/fgp)**|
|**2024-11-19**|**KDC-MAE: Knowledge Distilled Contrastive Mask Auto-Encoder**|Maheswar Bora et.al.|[2411.12270](http://arxiv.org/abs/2411.12270)|null|
|**2024-11-19**|**Just KIDDIN: Knowledge Infusion and Distillation for Detection of INdecent Memes**|Rahul Garg et.al.|[2411.12174](http://arxiv.org/abs/2411.12174)|null|
|**2024-11-18**|**Federated Incremental Named Entity Recognition**|Duzhen Zhang et.al.|[2411.11623](http://arxiv.org/abs/2411.11623)|null|
|**2024-11-18**|**Bridging the Resource Gap: Deploying Advanced Imitation Learning Models onto Affordable Embedded Platforms**|Haizhou Ge et.al.|[2411.11406](http://arxiv.org/abs/2411.11406)|null|
|**2024-11-17**|**Map-Free Trajectory Prediction with Map Distillation and Hierarchical Encoding**|Xiaodong Liu et.al.|[2411.10961](http://arxiv.org/abs/2411.10961)|null|
|**2024-11-16**|**Hybrid Attention Model Using Feature Decomposition and Knowledge Distillation for Glucose Forecasting**|Ebrahim Farahmand et.al.|[2411.10703](http://arxiv.org/abs/2411.10703)|null|
|**2024-11-16**|**Multi-perspective Contrastive Logit Distillation**|Qi Wang et.al.|[2411.10693](http://arxiv.org/abs/2411.10693)|null|
|**2024-11-16**|**Exploring Feature-based Knowledge Distillation For Recommender System: A Frequency Perspective**|Zhangchi Zhu et.al.|[2411.10676](http://arxiv.org/abs/2411.10676)|null|
|**2024-11-15**|**Scaling Law for Post-training after Model Pruning**|Xiaodong Chen et.al.|[2411.10272](http://arxiv.org/abs/2411.10272)|null|
|**2024-11-15**|**Evidential Federated Learning for Skin Lesion Image Classification**|Rutger Hendrix et.al.|[2411.10071](http://arxiv.org/abs/2411.10071)|null|
|**2024-11-14**|**VPBSD:Vessel-Pattern-Based Semi-Supervised Distillation for Efficient 3D Microscopic Cerebrovascular Segmentation**|Xi Lin et.al.|[2411.09567](http://arxiv.org/abs/2411.09567)|null|
|**2024-11-14**|**Re-Parameterization of Lightweight Transformer for On-Device Speech Emotion Recognition**|Zixing Zhang et.al.|[2411.09339](http://arxiv.org/abs/2411.09339)|null|
|**2024-11-14**|**Mono2Stereo: Monocular Knowledge Transfer for Enhanced Stereo Matching**|Yuran Wang et.al.|[2411.09151](http://arxiv.org/abs/2411.09151)|null|
|**2024-11-14**|**Toward Democratized Generative AI in Next-Generation Mobile Edge Networks**|Ruichen Zhang et.al.|[2411.09148](http://arxiv.org/abs/2411.09148)|null|
|**2024-11-13**|**Dual-Head Knowledge Distillation: Enhancing Logits Utilization with an Auxiliary Head**|Penghui Yang et.al.|[2411.08937](http://arxiv.org/abs/2411.08937)|null|
|**2024-11-13**|**UIFormer: A Unified Transformer-based Framework for Incremental Few-Shot Object Detection and Instance Segmentation**|Chengyuan Zhang et.al.|[2411.08569](http://arxiv.org/abs/2411.08569)|null|
|**2024-11-13**|**Federated Graph Learning with Graphless Clients**|Xingbo Fu et.al.|[2411.08374](http://arxiv.org/abs/2411.08374)|null|
|**2024-11-12**|**Joint Diffusion models in Continual Learning**|Paweł Skierś et.al.|[2411.08224](http://arxiv.org/abs/2411.08224)|null|
|**2024-11-12**|**Learning with Less: Knowledge Distillation from Large Language Models via Unlabeled Data**|Juanhui Li et.al.|[2411.08028](http://arxiv.org/abs/2411.08028)|null|
|**2024-11-13**|**Query Optimization for Parametric Knowledge Refinement in Retrieval-Augmented Large Language Models**|Youan Cong et.al.|[2411.07820](http://arxiv.org/abs/2411.07820)|null|
|**2024-11-12**|**ASER: Activation Smoothing and Error Reconstruction for Large Language Model Quantization**|Weibo Zhao et.al.|[2411.07762](http://arxiv.org/abs/2411.07762)|null|
|**2024-11-12**|**Optimizing Traffic Signal Control using High-Dimensional State Representation and Efficient Deep Reinforcement Learning**|Lawrence Francis et.al.|[2411.07759](http://arxiv.org/abs/2411.07759)|null|
|**2024-11-12**|**ALANINE: A Novel Decentralized Personalized Federated Learning For Heterogeneous LEO Satellite Constellation**|Liang Zhao et.al.|[2411.07752](http://arxiv.org/abs/2411.07752)|null|
|**2024-11-12**|**OWLed: Outlier-weighed Layerwise Pruning for Efficient Autonomous Driving Framework**|Jiaxi Li et.al.|[2411.07711](http://arxiv.org/abs/2411.07711)|**[link](https://github.com/JiaxiLi1/OWLed)**|
|**2024-11-13**|**Feature Interaction Fusion Self-Distillation Network For CTR Prediction**|Lei Sang et.al.|[2411.07508](http://arxiv.org/abs/2411.07508)|null|
|**2024-11-12**|**Quantifying Knowledge Distillation Using Partial Information Decomposition**|Pasan Dissanayake et.al.|[2411.07483](http://arxiv.org/abs/2411.07483)|null|
|**2024-11-11**|**SAMPart3D: Segment Any Part in 3D Objects**|Yunhan Yang et.al.|[2411.07184](http://arxiv.org/abs/2411.07184)|**[link](https://github.com/yhyang-myron/sampart3d-website)**|
|**2024-11-11**|**LLM-Neo: Parameter Efficient Knowledge Distillation for Large Language Models**|Runming Yang et.al.|[2411.06839](http://arxiv.org/abs/2411.06839)|null|
|**2024-11-11**|**ScaleKD: Strong Vision Transformers Could Be Excellent Teachers**|Jiawei Fan et.al.|[2411.06786](http://arxiv.org/abs/2411.06786)|**[link](https://github.com/deep-optimization/scalekd)**|
|**2024-11-11**|**An Efficient Memory Module for Graph Few-Shot Class-Incremental Learning**|Dong Li et.al.|[2411.06659](http://arxiv.org/abs/2411.06659)|**[link](https://github.com/arvin0313/mecoin-gfscil)**|
|**2024-11-10**|**CULL-MT: Compression Using Language and Layer pruning for Machine Translation**|Pedram Rostami et.al.|[2411.06506](http://arxiv.org/abs/2411.06506)|null|
|**2024-11-10**|**Over-parameterized Student Model via Tensor Decomposition Boosted Knowledge Distillation**|Yu-Liang Zhan et.al.|[2411.06448](http://arxiv.org/abs/2411.06448)|**[link](https://github.com/intell-sci-comput/opdf)**|
|**2024-11-09**|**Dynamic Textual Prompt For Rehearsal-free Lifelong Person Re-identification**|Hongyu Chen et.al.|[2411.06023](http://arxiv.org/abs/2411.06023)|null|
|**2024-11-09**|**Multi-hop RIS-aided Learning Model Sharing for Urban Air Mobility**|Kai Xiong et.al.|[2411.06015](http://arxiv.org/abs/2411.06015)|null|
|**2024-11-08**|**Mitigating Hallucination with ZeroG: An Advanced Knowledge Management Engine**|Anantha Sharma et.al.|[2411.05936](http://arxiv.org/abs/2411.05936)|null|
|**2024-11-08**|**Asterisk*: Keep it Simple**|Andrew Semenov et.al.|[2411.05691](http://arxiv.org/abs/2411.05691)|null|
|**2024-11-08**|**Knowledge Distillation Neural Network for Predicting Car-following Behaviour of Human-driven and Autonomous Vehicles**|Ayobami Adewale et.al.|[2411.05618](http://arxiv.org/abs/2411.05618)|null|
|**2024-11-08**|**Towards Lifelong Few-Shot Customization of Text-to-Image Diffusion**|Nan Song et.al.|[2411.05544](http://arxiv.org/abs/2411.05544)|null|
|**2024-11-07**|**ZipNN: Lossless Compression for AI Models**|Moshik Hershcovitch et.al.|[2411.05239](http://arxiv.org/abs/2411.05239)|**[link](https://github.com/zipnn/zipnn)**|
|**2024-11-07**|**Performance-Guided LLM Knowledge Distillation for Efficient Text Classification at Scale**|Flavio Di Palo et.al.|[2411.05045](http://arxiv.org/abs/2411.05045)|null|
|**2024-11-06**|**From Word Vectors to Multimodal Embeddings: Techniques, Applications, and Future Directions For Large Language Models**|Charles Zhang et.al.|[2411.05036](http://arxiv.org/abs/2411.05036)|null|
|**2024-11-07**|**Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers**|Zhichao Geng et.al.|[2411.04403](http://arxiv.org/abs/2411.04403)|null|
|**2024-11-07**|**GazeGen: Gaze-Driven User Interaction for Visual Content Generation**|He-Yen Hsieh et.al.|[2411.04335](http://arxiv.org/abs/2411.04335)|null|
|**2024-11-06**|**Towards Personalized Federated Learning via Comprehensive Knowledge Distillation**|Pengju Wang et.al.|[2411.03569](http://arxiv.org/abs/2411.03569)|null|
|**2024-11-05**|**Change Is the Only Constant: Dynamic LLM Slicing based on Layer Redundancy**|Razvan-Gabriel Dumitru et.al.|[2411.03513](http://arxiv.org/abs/2411.03513)|**[link](https://github.com/razvandu/dynamicslicing)**|
|**2024-11-05**|**Transformer-Based Fault-Tolerant Control for Fixed-Wing UAVs Using Knowledge Distillation and In-Context Adaptation**|Francisco Giral et.al.|[2411.02975](http://arxiv.org/abs/2411.02975)|null|
|**2024-11-05**|**Centerness-based Instance-aware Knowledge Distillation with Task-wise Mutual Lifting for Object Detection on Drone Imagery**|Bowei Du et.al.|[2411.02861](http://arxiv.org/abs/2411.02861)|null|
|**2024-11-05**|**Brewing Vodka: Distilling Pure Knowledge for Lightweight Threat Detection in Audit Logs**|Weiheng Wu et.al.|[2411.02775](http://arxiv.org/abs/2411.02775)|null|
|**2024-11-05**|**Multimodal Commonsense Knowledge Distillation for Visual Question Answering**|Shuo Yang et.al.|[2411.02722](http://arxiv.org/abs/2411.02722)|null|
|**2024-11-04**|**Information plane and compression-gnostic feedback in quantum machine learning**|Nathan Haboury et.al.|[2411.02313](http://arxiv.org/abs/2411.02313)|null|
|**2024-11-04**|**Training on the Test Model: Contamination in Ranking Distillation**|Vishakha Suresh Kalal et.al.|[2411.02284](http://arxiv.org/abs/2411.02284)|**[link](https://github.com/Parry-Parry/ContaminatedDistillation)**|
|**2024-11-03**|**Decoupling Dark Knowledge via Block-wise Logit Distillation for Feature-level Alignment**|Chengting Yu et.al.|[2411.01547](http://arxiv.org/abs/2411.01547)|null|
|**2024-11-01**|**On the Impact of White-box Deployment Strategies for Edge AI on Latency and Model Performance**|Jaskirat Singh et.al.|[2411.00907](http://arxiv.org/abs/2411.00907)|null|
|**2024-11-01**|**Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation**|Bohan Lyu et.al.|[2411.00412](http://arxiv.org/abs/2411.00412)|null|
|**2024-11-01**|**Towards Building Secure UAV Navigation with FHE-aware Knowledge Distillation**|Arjun Ramesh Kaushik et.al.|[2411.00403](http://arxiv.org/abs/2411.00403)|null|
|**2024-11-01**|**Efficient Model Compression for Bayesian Neural Networks**|Diptarka Saha et.al.|[2411.00273](http://arxiv.org/abs/2411.00273)|null|
|**2024-10-31**|**Semantic Knowledge Distillation for Onboard Satellite Earth Observation Image Classification**|Thanh-Dung Le et.al.|[2411.00209](http://arxiv.org/abs/2411.00209)|**[link](https://github.com/ltdung/snt-sentry)**|
|**2024-10-31**|**Mutual Information Preserving Neural Network Pruning**|Charles Westphal et.al.|[2411.00147](http://arxiv.org/abs/2411.00147)|null|
|**2024-10-30**|**Larger models yield better results? Streamlined severity classification of ADHD-related concerns using BERT-based knowledge distillation**|Ahmed Akib Jawad Karim et.al.|[2411.00052](http://arxiv.org/abs/2411.00052)|null|
|**2024-10-30**|**IP-MOT: Instance Prompt Learning for Cross-Domain Multi-Object Tracking**|Run Luo et.al.|[2410.23907](http://arxiv.org/abs/2410.23907)|null|
|**2024-10-29**|**ML Research Benchmark**|Matthew Kenney et.al.|[2410.22553](http://arxiv.org/abs/2410.22553)|**[link](https://github.com/algorithmicresearchgroup/ml-research-agent)**|
|**2024-11-01**|**Leveraging Recurrent Neural Networks for Predicting Motor Movements from Primate Motor Cortex Neural Recordings**|Yuanxi Wang et.al.|[2410.22283](http://arxiv.org/abs/2410.22283)|null|
|**2024-10-28**|**Unveiling Context-Aware Criteria in Self-Assessing LLMs**|Taneesh Gupta et.al.|[2410.21545](http://arxiv.org/abs/2410.21545)|null|
|**2024-10-28**|**Knowledge Distillation for Real-Time Classification of Early Media in Voice Communications**|Kemal Altwlkany et.al.|[2410.21478](http://arxiv.org/abs/2410.21478)|null|
|**2024-10-31**|**LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment**|Ge Yang et.al.|[2410.21352](http://arxiv.org/abs/2410.21352)|**[link](https://github.com/aboveparadise/llmcbench)**|
|**2024-10-28**|**EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation**|Shih-Yang Liu et.al.|[2410.21271](http://arxiv.org/abs/2410.21271)|null|
|**2024-10-28**|**Deep Learning for Medical Text Processing: BERT Model Fine-Tuning and Comparative Study**|Jiacheng Hu et.al.|[2410.20792](http://arxiv.org/abs/2410.20792)|null|
|**2024-10-28**|**KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation**|Rambod Azimi et.al.|[2410.20777](http://arxiv.org/abs/2410.20777)|**[link](https://github.com/rambodazimi/kd-lora)**|
|**2024-10-28**|**Data-Efficient Low-Complexity Acoustic Scene Classification via Distilling and Progressive Pruning**|Bing Han et.al.|[2410.20775](http://arxiv.org/abs/2410.20775)|null|
|**2024-10-28**|**Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA**|Sangmin Bae et.al.|[2410.20672](http://arxiv.org/abs/2410.20672)|null|
|**2024-10-27**|**Uncovering Capabilities of Model Pruning in Graph Contrastive Learning**|Wu Junran et.al.|[2410.20356](http://arxiv.org/abs/2410.20356)|null|
|**2024-10-25**|**A Survey of Small Language Models**|Chien Van Nguyen et.al.|[2410.20011](http://arxiv.org/abs/2410.20011)|null|
|**2024-10-25**|**GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing**|Hosam Elgendy et.al.|[2410.19552](http://arxiv.org/abs/2410.19552)|**[link](https://github.com/HosamGen/GeoLLaVA)**|
|**2024-10-25**|**SWITCH: Studying with Teacher for Knowledge Distillation of Large Language Models**|Jahyun Koo et.al.|[2410.19503](http://arxiv.org/abs/2410.19503)|null|
|**2024-10-24**|**Tailored-LLaMA: Optimizing Few-Shot Learning in Pruned LLaMA Models with Task-Specific Prompts**|Danyal Aftab et.al.|[2410.19185](http://arxiv.org/abs/2410.19185)|null|
|**2024-10-24**|**AlignCap: Aligning Speech Emotion Captioning to Human Preferences**|Ziqi Liang et.al.|[2410.19134](http://arxiv.org/abs/2410.19134)|null|
|**2024-10-24**|**High-dimensional Analysis of Knowledge Distillation: Weak-to-Strong Generalization and Scaling Laws**|M. Emrullah Ildiz et.al.|[2410.18837](http://arxiv.org/abs/2410.18837)|null|
|**2024-10-24**|**Knowledge Distillation Using Frontier Open-source LLMs: Generalizability and the Role of Synthetic Data**|Anup Shirgaonkar et.al.|[2410.18588](http://arxiv.org/abs/2410.18588)|null|
|**2024-10-24**|**SIKeD: Self-guided Iterative Knowledge Distillation for mathematical reasoning**|Shivam Adarsh et.al.|[2410.18574](http://arxiv.org/abs/2410.18574)|**[link](https://github.com/kumar-shridhar/siked)**|
|**2024-10-23**|**ELAICHI: Enhancing Low-resource TTS by Addressing Infrequent and Low-frequency Character Bigrams**|Srija Anand et.al.|[2410.17901](http://arxiv.org/abs/2410.17901)|null|
|**2024-10-23**|**Beware of Calibration Data for Pruning Large Language Models**|Yixin Ji et.al.|[2410.17711](http://arxiv.org/abs/2410.17711)|null|
|**2024-10-23**|**Towards Active Participant-Centric Vertical Federated Learning: Some Representations May Be All You Need**|Jon Irureta et.al.|[2410.17648](http://arxiv.org/abs/2410.17648)|null|
|**2024-10-23**|**Towards Effective Data-Free Knowledge Distillation via Diverse Diffusion Augmentation**|Muquan Li et.al.|[2410.17606](http://arxiv.org/abs/2410.17606)|**[link](https://github.com/slgsp/dda)**|
|**2024-10-23**|**Multimodal Information Bottleneck for Deep Reinforcement Learning with Multiple Sensors**|Bang You et.al.|[2410.17551](http://arxiv.org/abs/2410.17551)|null|
|**2024-10-23**|**Physics-driven AI for Channel Estimation in Cellular Network**|Xiaoqian Qi et.al.|[2410.17525](http://arxiv.org/abs/2410.17525)|null|
|**2024-10-22**|**MiniPLM: Knowledge Distillation for Pre-Training Language Models**|Yuxian Gu et.al.|[2410.17215](http://arxiv.org/abs/2410.17215)|**[link](https://github.com/thu-coai/miniplm)**|
|**2024-10-22**|**Self-calibration for Language Model Quantization and Pruning**|Miles Williams et.al.|[2410.17170](http://arxiv.org/abs/2410.17170)|null|
|**2024-10-22**|**DiP-GO: A Diffusion Pruner via Few-step Gradient Optimization**|Haowei Zhu et.al.|[2410.16942](http://arxiv.org/abs/2410.16942)|null|
|**2024-10-22**|**Mitigating Vanishing Activations in Deep CapsNets Using Channel Pruning**|Siddharth Sahu et.al.|[2410.16908](http://arxiv.org/abs/2410.16908)|**[link](https://github.com/sidsahu88/leeds_msc_ocom5300m_project)**|
|**2024-10-22**|**CK4Gen: A Knowledge Distillation Framework for Generating High-Utility Synthetic Survival Datasets in Healthcare**|Nicholas I-Hsien Kuo et.al.|[2410.16872](http://arxiv.org/abs/2410.16872)|null|
|**2024-10-22**|**AttriPrompter: Auto-Prompting with Attribute Semantics for Zero-shot Nuclei Detection via Visual-Language Pre-trained Models**|Yongjian Wu et.al.|[2410.16820](http://arxiv.org/abs/2410.16820)|**[link](https://github.com/wuyongjiancode/attriprompter)**|
|**2024-10-22**|**SafetyAnalyst: Interpretable, transparent, and steerable LLM safety moderation**|Jing-Jing Li et.al.|[2410.16665](http://arxiv.org/abs/2410.16665)|null|
|**2024-10-21**|**Pre-training Distillation for Large Language Models: A Design Space Exploration**|Hao Peng et.al.|[2410.16215](http://arxiv.org/abs/2410.16215)|null|
|**2024-10-18**|**Interpreting Microbiome Relative Abundance Data Using Symbolic Regression**|Swagatam Haldar et.al.|[2410.16109](http://arxiv.org/abs/2410.16109)|**[link](https://github.com/swag2198/microbiome-symbolic-regression)**|
|**2024-10-21**|**Model Mimic Attack: Knowledge Distillation for Provably Transferable Adversarial Examples**|Kirill Lukyanov et.al.|[2410.15889](http://arxiv.org/abs/2410.15889)|null|
|**2024-10-20**|**GSSF: Generalized Structural Sparse Function for Deep Cross-modal Metric Learning**|Haiwen Diao et.al.|[2410.15266](http://arxiv.org/abs/2410.15266)|**[link](https://github.com/paranioar/gssf)**|
|**2024-10-19**|**LLaVA-Ultra: Large Chinese Language and Vision Assistant for Ultrasound**|Xuechen Guo et.al.|[2410.15074](http://arxiv.org/abs/2410.15074)|null|
|**2024-10-19**|**Improving Pronunciation and Accent Conversion through Knowledge Distillation And Synthetic Ground-Truth from Native TTS**|Tuan Nam Nguyen et.al.|[2410.14997](http://arxiv.org/abs/2410.14997)|null|
|**2024-10-18**|**EvoPress: Towards Optimal Dynamic Model Compression via Evolutionary Search**|Oliver Sieberling et.al.|[2410.14649](http://arxiv.org/abs/2410.14649)|**[link](https://github.com/ist-daslab/evopress)**|
|**2024-10-18**|**Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation**|Shuai Zhao et.al.|[2410.14425](http://arxiv.org/abs/2410.14425)|**[link](https://github.com/shuaizhao95/Unlearning)**|
|**2024-10-18**|**Preview-based Category Contrastive Learning for Knowledge Distillation**|Muhe Ding et.al.|[2410.14143](http://arxiv.org/abs/2410.14143)|null|
|**2024-10-17**|**Leveraging Fine-Tuned Language Models for Efficient and Accurate Smart Contract Auditing**|Zhiyuan Wei et.al.|[2410.13918](http://arxiv.org/abs/2410.13918)|**[link](https://github.com/LLMSmartAudit/FTSmartAudit)**|
|**2024-10-17**|**An Active Learning Framework for Inclusive Generation by Large Language Models**|Sabit Hassan et.al.|[2410.13641](http://arxiv.org/abs/2410.13641)|null|
|**2024-10-18**|**Towards Satellite Non-IID Imagery: A Spectral Clustering-Assisted Federated Learning Approach**|Luyao Zou et.al.|[2410.13602](http://arxiv.org/abs/2410.13602)|null|
|**2024-10-18**|**Cyber Attacks Prevention Towards Prosumer-based EV Charging Stations: An Edge-assisted Federated Prototype Knowledge Distillation Approach**|Luyao Zou et.al.|[2410.13260](http://arxiv.org/abs/2410.13260)|null|
|**2024-10-16**|**TAS: Distilling Arbitrary Teacher and Student via a Hybrid Assistant**|Guopeng Li et.al.|[2410.12342](http://arxiv.org/abs/2410.12342)|null|
|**2024-10-16**|**Optimizing YOLOv5s Object Detection through Knowledge Distillation algorithm**|Guanming Huang et.al.|[2410.12259](http://arxiv.org/abs/2410.12259)|null|
|**2024-10-16**|**TransAgent: Transfer Vision-Language Foundation Models with Heterogeneous Agent Collaboration**|Yiwei Guo et.al.|[2410.12183](http://arxiv.org/abs/2410.12183)|**[link](https://github.com/markywg/transagent)**|
|**2024-10-17**|**SAM-Guided Masked Token Prediction for 3D Scene Understanding**|Zhimin Chen et.al.|[2410.12158](http://arxiv.org/abs/2410.12158)|null|
|**2024-10-15**|**MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router**|Yanyue Xie et.al.|[2410.12013](http://arxiv.org/abs/2410.12013)|null|
|**2024-10-15**|**Breaking Modality Gap in RGBT Tracking: Coupled Knowledge Distillation**|Andong Lu et.al.|[2410.11586](http://arxiv.org/abs/2410.11586)|**[link](https://github.com/multi-modality-tracking/ckd)**|
|**2024-10-15**|**Learning from Imperfect Data: Towards Efficient Knowledge Distillation of Autoregressive Language Models for Text-to-SQL**|Qihuang Zhong et.al.|[2410.11371](http://arxiv.org/abs/2410.11371)|null|
|**2024-10-15**|**Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling**|Wenda Xu et.al.|[2410.11325](http://arxiv.org/abs/2410.11325)|null|
|**2024-10-14**|**ROSAR: An Adversarial Re-Training Framework for Robust Side-Scan Sonar Object Detection**|Martin Aubard et.al.|[2410.10554](http://arxiv.org/abs/2410.10554)|**[link](https://github.com/remaro-network/rosar-framework)**|
|**2024-10-14**|**QIANets: Quantum-Integrated Adaptive Networks for Reduced Latency and Improved Inference Times in CNN Models**|Zhumazhan Balapanov et.al.|[2410.10318](http://arxiv.org/abs/2410.10318)|**[link](https://github.com/edwardmagongo/quantum-inspired-model-compression)**|
|**2024-10-14**|**Temperature-Centric Investigation of Speculative Decoding with Knowledge Distillation**|Siru Ouyang et.al.|[2410.10141](http://arxiv.org/abs/2410.10141)|null|
|**2024-10-15**|**Edge Unlearning is Not "on Edge"! An Adaptive Exact Unlearning System on Resource-Constrained Devices**|Xiaoyu Xia et.al.|[2410.10128](http://arxiv.org/abs/2410.10128)|**[link](https://github.com/xlab-hub/cause)**|
|**2024-10-14**|**REHRSeg: Unleashing the Power of Self-Supervised Super-Resolution for Resource-Efficient 3D MRI Segmentation**|Zhiyun Song et.al.|[2410.10097](http://arxiv.org/abs/2410.10097)|null|
|**2024-10-12**|**SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs**|Mohammad Mozaffari et.al.|[2410.09615](http://arxiv.org/abs/2410.09615)|**[link](https://github.com/mohammad-mozaffari/slim)**|
|**2024-10-12**|**Distilling Invariant Representations with Dual Augmentation**|Nikolaos Giakoumoglou et.al.|[2410.09474](http://arxiv.org/abs/2410.09474)|null|
|**2024-10-12**|**Declarative Knowledge Distillation from Large Language Models for Visual Question Answering Datasets**|Thomas Eiter et.al.|[2410.09428](http://arxiv.org/abs/2410.09428)|**[link](https://github.com/pudumagico/kr2024)**|
|**2024-10-15**|**Transforming In-Vehicle Network Intrusion Detection: VAE-based Knowledge Distillation Meets Explainable AI**|Muhammet Anil Yagiz et.al.|[2410.09043](http://arxiv.org/abs/2410.09043)|null|
|**2024-10-11**|**Mentor-KD: Making Small Language Models Better Multi-step Reasoners**|Hojae Lee et.al.|[2410.09037](http://arxiv.org/abs/2410.09037)|**[link](https://github.com/2hojae/mentor-kd)**|
|**2024-10-11**|**Contrastive Knowledge Distillation for Robust Multimodal Sentiment Analysis**|Zhongyi Sang et.al.|[2410.08692](http://arxiv.org/abs/2410.08692)|null|
|**2024-10-11**|**GAI-Enabled Explainable Personalized Federated Semi-Supervised Learning**|Yubo Peng et.al.|[2410.08634](http://arxiv.org/abs/2410.08634)|null|
|**2024-10-11**|**Simultaneous Reward Distillation and Preference Learning: Get You a Language Model Who Can Do Both**|Abhijnan Nath et.al.|[2410.08458](http://arxiv.org/abs/2410.08458)|null|
|**2024-10-10**|**What is Left After Distillation? How Knowledge Transfer Impacts Fairness and Bias**|Aida Mohammadshahi et.al.|[2410.08407](http://arxiv.org/abs/2410.08407)|null|
|**2024-10-10**|**Non-transferable Pruning**|Ruyi Ding et.al.|[2410.08015](http://arxiv.org/abs/2410.08015)|null|
|**2024-10-10**|**A Lightweight Target-Driven Network of Stereo Matching for Inland Waterways**|Jing Su et.al.|[2410.07915](http://arxiv.org/abs/2410.07915)|null|
|**2024-10-10**|**SNN-PAR: Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks**|Haiyang Wang et.al.|[2410.07857](http://arxiv.org/abs/2410.07857)|**[link](https://github.com/event-ahu/openpar)**|
|**2024-10-12**|**Relational Diffusion Distillation for Efficient Image Generation**|Weilun Feng et.al.|[2410.07679](http://arxiv.org/abs/2410.07679)|**[link](https://github.com/cantbebetter2/rdd)**|
|**2024-10-10**|**CrossQuant: A Post-Training Quantization Method with Smaller Quantization Kernel for Precise Large Language Model Compression**|Wenyuan Liu et.al.|[2410.07505](http://arxiv.org/abs/2410.07505)|null|
|**2024-10-09**|**Unlocking Real-Time Fluorescence Lifetime Imaging: Multi-Pixel Parallelism for FPGA-Accelerated Processing**|Ismail Erbas et.al.|[2410.07364](http://arxiv.org/abs/2410.07364)|null|
|**2024-10-09**|**S2HPruner: Soft-to-Hard Distillation Bridges the Discretization Gap in Pruning**|Weihao Lin et.al.|[2410.07046](http://arxiv.org/abs/2410.07046)|null|
|**2024-10-09**|**Structure-Centric Robust Monocular Depth Estimation via Knowledge Distillation**|Runze Chen et.al.|[2410.06982](http://arxiv.org/abs/2410.06982)|null|
|**2024-10-09**|**Efficient and Robust Knowledge Distillation from A Stronger Teacher Based on Correlation Matching**|Wenqi Niu et.al.|[2410.06561](http://arxiv.org/abs/2410.06561)|null|
|**2024-10-08**|**SpaLLM: Unified Compressive Adaptation of Large Language Models with Sketching**|Tianyi Zhang et.al.|[2410.06364](http://arxiv.org/abs/2410.06364)|null|
|**2024-10-08**|**QT-DoG: Quantization-aware Training for Domain Generalization**|Saqib Javed et.al.|[2410.06020](http://arxiv.org/abs/2410.06020)|**[link](https://github.com/saqibjaved1/QT-DoG)**|
|**2024-10-10**|**KnowledgeSG: Privacy-Preserving Synthetic Text Generation with Knowledge Distillation from Server**|Wenhao Wang et.al.|[2410.05725](http://arxiv.org/abs/2410.05725)|**[link](https://github.com/wwh0411/knowledgesg)**|
|**2024-10-07**|**Progressive distillation induces an implicit curriculum**|Abhishek Panigrahi et.al.|[2410.05464](http://arxiv.org/abs/2410.05464)|null|
|**2024-10-07**|**ESPACE: Dimensionality Reduction of Activations for Model Compression**|Charbel Sakr et.al.|[2410.05437](http://arxiv.org/abs/2410.05437)|null|
|**2024-10-07**|**ReasoningRank: Teaching Student Models to Rank through Reasoning-Based Knowledge Distillation**|Yuelyu Ji et.al.|[2410.05168](http://arxiv.org/abs/2410.05168)|null|
|**2024-10-06**|**CAPEEN: Image Captioning with Early Exits and Knowledge Distillation**|Divya Jyoti Bajpai et.al.|[2410.04433](http://arxiv.org/abs/2410.04433)|**[link](https://github.com/div290/capeen)**|
|**2024-10-06**|**DAdEE: Unsupervised Domain Adaptation in Early Exit PLMs**|Divya Jyoti Bajpai et.al.|[2410.04424](http://arxiv.org/abs/2410.04424)|**[link](https://github.com/div290/dadee)**|
|**2024-10-05**|**Distillation-Free One-Step Diffusion for Real-World Image Super-Resolution**|Jianze Li et.al.|[2410.04224](http://arxiv.org/abs/2410.04224)|**[link](https://github.com/jianzeli-114/dfosd)**|
|**2024-10-05**|**Accelerating Diffusion Models with One-to-Many Knowledge Distillation**|Linfeng Zhang et.al.|[2410.04191](http://arxiv.org/abs/2410.04191)|null|
|**2024-10-05**|**DiDOTS: Knowledge Distillation from Large-Language-Models for Dementia Obfuscation in Transcribed Speech**|Dominika Woszczyk et.al.|[2410.04188](http://arxiv.org/abs/2410.04188)|null|
|**2024-10-05**|**Gap Preserving Distillation by Building Bidirectional Mappings with A Dynamic Teacher**|Yong Guo et.al.|[2410.04140](http://arxiv.org/abs/2410.04140)|null|
|**2024-10-04**|**Enhance Reasoning by Learning from Mistakes: Peer-Review Knowledge Distillation from Multiple Large Language Models**|Zhuochun Li et.al.|[2410.03663](http://arxiv.org/abs/2410.03663)|null|
|**2024-10-04**|**DocKD: Knowledge Distillation from LLMs for Open-World Document Understanding Models**|Sungnyun Kim et.al.|[2410.03061](http://arxiv.org/abs/2410.03061)|null|
|**2024-10-03**|**Geometry is All You Need: A Unified Taxonomy of Matrix and Tensor Factorization for Compression of Generative Language Models**|Mingxue Xu et.al.|[2410.03040](http://arxiv.org/abs/2410.03040)|null|
|**2024-10-03**|**Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-Training of Deep Networks**|Siddharth Joshi et.al.|[2410.02116](http://arxiv.org/abs/2410.02116)|null|
|**2024-10-02**|**Review Non-convex Optimization Method for Machine Learning**|Greg B Fotopoulos et.al.|[2410.02017](http://arxiv.org/abs/2410.02017)|null|
|**2024-10-02**|**PHI-S: Distribution Balancing for Label-Free Multi-Teacher Distillation**|Mike Ranzinger et.al.|[2410.01680](http://arxiv.org/abs/2410.01680)|null|
|**2024-10-04**|**HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models**|Seanie Lee et.al.|[2410.01524](http://arxiv.org/abs/2410.01524)|**[link](https://github.com/imnotkind/HarmAug)**|
|**2024-10-02**|**Foldable SuperNets: Scalable Merging of Transformers with Different Initializations and Tasks**|Edan Kinderman et.al.|[2410.01483](http://arxiv.org/abs/2410.01483)|**[link](https://github.com/idankinderman/fs_merge)**|
|**2024-10-02**|**PairDistill: Pairwise Relevance Distillation for Dense Retrieval**|Chao-Wei Huang et.al.|[2410.01383](http://arxiv.org/abs/2410.01383)|**[link](https://github.com/miulab/pairdistill)**|
|**2024-10-02**|**"No Matter What You Do!": Mitigating Backdoor Attacks in Graph Neural Networks**|Jiale Zhang et.al.|[2410.01272](http://arxiv.org/abs/2410.01272)|**[link](https://github.com/graph-axis/gcleaner)**|
|**2024-10-01**|**Compressing Recurrent Neural Networks for FPGA-accelerated Implementation in Fluorescence Lifetime Imaging**|Ismail Erbas et.al.|[2410.00948](http://arxiv.org/abs/2410.00948)|null|
|**2024-10-01**|**Local-to-Global Self-Supervised Representation Learning for Diabetic Retinopathy Grading**|Mostafa Hajighasemloua et.al.|[2410.00779](http://arxiv.org/abs/2410.00779)|null|
|**2024-10-01**|**Efficient Technical Term Translation: A Knowledge Distillation Approach for Parenthetical Terminology Translation**|Jiyoon Myung et.al.|[2410.00683](http://arxiv.org/abs/2410.00683)|null|
|**2024-10-01**|**AMR-Evol: Adaptive Modular Response Evolution Elicits Better Knowledge Distillation for Large Language Models in Code Generation**|Ziyang Luo et.al.|[2410.00558](http://arxiv.org/abs/2410.00558)|**[link](https://github.com/chiyeunglaw/amr-evol)**|
|**2024-10-01**|**Self-Updatable Large Language Models with Parameter Integration**|Yu Wang et.al.|[2410.00487](http://arxiv.org/abs/2410.00487)|null|
|**2024-09-30**|**Enhancing Romanian Offensive Language Detection through Knowledge Distillation, Multi-Task Learning, and Data Augmentation**|Vlad-Cristian Matei et.al.|[2409.20498](http://arxiv.org/abs/2409.20498)|null|
|**2024-10-02**|**Linear Projections of Teacher Embeddings for Few-Class Distillation**|Noel Loo et.al.|[2409.20449](http://arxiv.org/abs/2409.20449)|null|
|**2024-09-30**|**Classroom-Inspired Multi-Mentor Distillation with Adaptive Learning Strategies**|Shalini Sarode et.al.|[2409.20237](http://arxiv.org/abs/2409.20237)|null|
|**2024-09-30**|**Aggressive Post-Training Compression on Extremely Large Language Models**|Zining Zhang et.al.|[2409.20094](http://arxiv.org/abs/2409.20094)|null|
|**2024-10-01**|**HYDRA-FL: Hybrid Knowledge Distillation for Robust and Accurate Federated Learning**|Momin Ahmad Khan et.al.|[2409.19912](http://arxiv.org/abs/2409.19912)|null|
|**2024-09-29**|**Tailored Federated Learning: Leveraging Direction Regulation & Knowledge Distillation**|Huidong Tang et.al.|[2409.19741](http://arxiv.org/abs/2409.19741)|null|
|**2024-09-29**|**InfantCryNet: A Data-driven Framework for Intelligent Analysis of Infant Cries**|Mengze Hong et.al.|[2409.19689](http://arxiv.org/abs/2409.19689)|null|
|**2024-09-28**|**Value-Based Deep Multi-Agent Reinforcement Learning with Dynamic Sparse Training**|Pihe Hu et.al.|[2409.19391](http://arxiv.org/abs/2409.19391)|null|
|**2024-09-28**|**Mind the Gap: Promoting Missing Modality Brain Tumor Segmentation with Alignment**|Tianyi Liu et.al.|[2409.19366](http://arxiv.org/abs/2409.19366)|null|
|**2024-09-27**|**Semi-Supervised Bone Marrow Lesion Detection from Knee MRI Segmentation Using Mask Inpainting Models**|Shihua Qin et.al.|[2409.19185](http://arxiv.org/abs/2409.19185)|null|
|**2024-09-27**|**MiniVLN: Efficient Vision-and-Language Navigation by Progressive Knowledge Distillation**|Junyou Zhu et.al.|[2409.18800](http://arxiv.org/abs/2409.18800)|null|
|**2024-09-27**|**Student-Oriented Teacher Knowledge Refinement for Knowledge Distillation**|Chaomin Shen et.al.|[2409.18785](http://arxiv.org/abs/2409.18785)|null|
|**2024-09-27**|**Harmonizing knowledge Transfer in Neural Network with Unified Distillation**|Yaomin Huang et.al.|[2409.18565](http://arxiv.org/abs/2409.18565)|null|
|**2024-09-27**|**Towards Diverse Device Heterogeneous Federated Learning via Task Arithmetic Knowledge Integration**|Mahdi Morafah et.al.|[2409.18461](http://arxiv.org/abs/2409.18461)|**[link](https://github.com/mmorafah/takfl)**|
|**2024-09-26**|**EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation**|Jiaxiang Tang et.al.|[2409.18114](http://arxiv.org/abs/2409.18114)|null|
|**2024-09-26**|**Weak-To-Strong Backdoor Attacks for LLMs with Contrastive Knowledge Distillation**|Shuai Zhao et.al.|[2409.17946](http://arxiv.org/abs/2409.17946)|null|
|**2024-09-26**|**Kendall's $τ$ Coefficient for Logits Distillation**|Yuchen Guan et.al.|[2409.17823](http://arxiv.org/abs/2409.17823)|null|
|**2024-09-26**|**General Compression Framework for Efficient Transformer Object Tracking**|Lingyi Hong et.al.|[2409.17564](http://arxiv.org/abs/2409.17564)|null|
|**2024-09-26**|**Shape-intensity knowledge distillation for robust medical image segmentation**|Wenhui Dong et.al.|[2409.17503](http://arxiv.org/abs/2409.17503)|**[link](https://github.com/whdong-whu/sikd)**|
|**2024-09-25**|**Search for Efficient Large Language Models**|Xuan Shen et.al.|[2409.17372](http://arxiv.org/abs/2409.17372)|**[link](https://github.com/shawnricecake/search-llm)**|
|**2024-09-25**|**MT2KD: Towards A General-Purpose Encoder for Speech, Speaker, and Audio Events**|Xiaoyu Yang et.al.|[2409.17010](http://arxiv.org/abs/2409.17010)|null|
|**2024-09-25**|**Adverse Weather Optical Flow: Cumulative Homogeneous-Heterogeneous Adaptation**|Hanyu Zhou et.al.|[2409.17001](http://arxiv.org/abs/2409.17001)|null|
|**2024-09-25**|**SelectiveKD: A semi-supervised framework for cancer detection in DBT through Knowledge Distillation and Pseudo-labeling**|Laurent Dillard et.al.|[2409.16581](http://arxiv.org/abs/2409.16581)|null|
|**2024-09-24**|**AIM 2024 Challenge on UHD Blind Photo Quality Assessment**|Vlad Hosu et.al.|[2409.16271](http://arxiv.org/abs/2409.16271)|null|
|**2024-09-25**|**Privacy Evaluation Benchmarks for NLP Models**|Wei Huang et.al.|[2409.15868](http://arxiv.org/abs/2409.15868)|**[link](https://github.com/user2311717757/nlp_doctor)**|
|**2024-09-24**|**Twin Network Augmentation: A Novel Training Strategy for Improved Spiking Neural Networks and Efficient Weight Quantization**|Lucas Deckers et.al.|[2409.15849](http://arxiv.org/abs/2409.15849)|null|
|**2024-09-23**|**TS-TCD: Triplet-Level Cross-Modal Distillation for Time-Series Forecasting Using Large Language Models**|Pengfei Wang et.al.|[2409.14978](http://arxiv.org/abs/2409.14978)|null|
|**2024-09-23**|**DSG-KD: Knowledge Distillation from Domain-Specific to General Language Models**|Sangyeon Cho et.al.|[2409.14904](http://arxiv.org/abs/2409.14904)|**[link](https://github.com/josangyeon/dsg-kd)**|
|**2024-09-23**|**Pre-trained Language Model and Knowledge Distillation for Lightweight Sequential Recommendation**|Li Li et.al.|[2409.14810](http://arxiv.org/abs/2409.14810)|null|
|**2024-09-23**|**An Adverse Weather-Immune Scheme with Unfolded Regularization and Foundation Model Knowledge Distillation for Street Scene Understanding**|Wei-Bin Kou et.al.|[2409.14737](http://arxiv.org/abs/2409.14737)|null|
|**2024-09-18**|**Applications of Knowledge Distillation in Remote Sensing: A Survey**|Yassine Himeur et.al.|[2409.12111](http://arxiv.org/abs/2409.12111)|null|
|**2024-09-18**|**Data Efficient Acoustic Scene Classification using Teacher-Informed Confusing Class Instruction**|Jin Jie Sean Yeo et.al.|[2409.11964](http://arxiv.org/abs/2409.11964)|null|
|**2024-09-18**|**Distillation-free Scaling of Large SSMs for Images and Videos**|Hamid Suleman et.al.|[2409.11867](http://arxiv.org/abs/2409.11867)|null|
|**2024-09-18**|**EFCM: Efficient Fine-tuning on Compressed Models for deployment of large models in medical image analysis**|Shaojie Li et.al.|[2409.11817](http://arxiv.org/abs/2409.11817)|null|
|**2024-09-18**|**RUIE: Retrieval-based Unified Information Extraction using Large Language Model**|Xincheng Liao et.al.|[2409.11673](http://arxiv.org/abs/2409.11673)|null|
|**2024-09-17**|**Time-Series Forecasting, Knowledge Distillation, and Refinement within a Multimodal PDE Foundation Model**|Derek Jollie et.al.|[2409.11609](http://arxiv.org/abs/2409.11609)|**[link](https://github.com/jingminsun/prose_v1)**|
|**2024-09-17**|**Unleashing the Potential of Mamba: Boosting a LiDAR 3D Sparse Detector by Using Cross-Model Knowledge Distillation**|Rui Yu et.al.|[2409.11018](http://arxiv.org/abs/2409.11018)|null|
|**2024-09-17**|**Single-stage TTS with Masked Audio Token Modeling and Semantic Knowledge Distillation**|Gerard I. Gállego et.al.|[2409.11003](http://arxiv.org/abs/2409.11003)|null|
|**2024-09-16**|**Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning**|Amin Karimi Monsefi et.al.|[2409.10362](http://arxiv.org/abs/2409.10362)|null|
|**2024-09-16**|**Human Insights Driven Latent Space for Different Driving Perspectives: A Unified Encoder for Efficient Multi-Task Inference**|Huy-Dung Nguyen et.al.|[2409.10095](http://arxiv.org/abs/2409.10095)|null|
|**2024-09-15**|**ELSA: Exploiting Layer-wise N:M Sparsity for Vision Transformer Acceleration**|Ning-Chi Huang et.al.|[2409.09708](http://arxiv.org/abs/2409.09708)|null|
|**2024-09-14**|**Effective Pre-Training of Audio Transformers for Sound Event Detection**|Florian Schmid et.al.|[2409.09546](http://arxiv.org/abs/2409.09546)|**[link](https://github.com/fschmid56/pretrainedsed)**|
|**2024-09-14**|**Integrated Multi-Level Knowledge Distillation for Enhanced Speaker Verification**|Wenhao Yang et.al.|[2409.09389](http://arxiv.org/abs/2409.09389)|null|
|**2024-09-14**|**Joint Semantic Knowledge Distillation and Masked Acoustic Modeling for Full-band Speech Restoration with Improved Intelligibility**|Xiaoyu Liu et.al.|[2409.09357](http://arxiv.org/abs/2409.09357)|null|
|**2024-09-13**|**Exploring System-Heterogeneous Federated Learning with Dynamic Model Selection**|Dixi Yao et.al.|[2409.08858](http://arxiv.org/abs/2409.08858)|null|
|**2024-09-13**|**An Efficient Privacy-aware Split Learning Framework for Satellite Communications**|Jianfei Sun et.al.|[2409.08538](http://arxiv.org/abs/2409.08538)|null|
|**2024-09-13**|**AWF: Adaptive Weight Fusion for Enhanced Class Incremental Semantic Segmentation**|Zechao Sun et.al.|[2409.08516](http://arxiv.org/abs/2409.08516)|null|
|**2024-09-12**|**DiReDi: Distillation and Reverse Distillation for AIoT Applications**|Chen Sun et.al.|[2409.08308](http://arxiv.org/abs/2409.08308)|null|
|**2024-09-12**|**Ruri: Japanese General Text Embeddings**|Hayato Tsukagoshi et.al.|[2409.07737](http://arxiv.org/abs/2409.07737)|**[link](https://github.com/oshizo/japaneseembeddingeval)**|
|**2024-09-12**|**Learn from Balance: Rectifying Knowledge Transfer for Long-Tailed Scenarios**|Xinlei Huang et.al.|[2409.07694](http://arxiv.org/abs/2409.07694)|null|
|**2024-09-11**|**DS-ViT: Dual-Stream Vision Transformer for Cross-Task Distillation in Alzheimer's Early Diagnosis**|Ke Chen et.al.|[2409.07584](http://arxiv.org/abs/2409.07584)|null|
|**2024-09-11**|**EchoDFKD: Data-Free Knowledge Distillation for Cardiac Ultrasound Segmentation using Synthetic Data**|Grégoire Petit et.al.|[2409.07566](http://arxiv.org/abs/2409.07566)|null|
|**2024-09-11**|**NVRC: Neural Video Representation Compression**|Ho Man Kwan et.al.|[2409.07414](http://arxiv.org/abs/2409.07414)|null|
|**2024-09-11**|**Enhancing CTC-Based Visual Speech Recognition**|Hendrik Laux et.al.|[2409.07210](http://arxiv.org/abs/2409.07210)|null|
|**2024-09-11**|**A Continual and Incremental Learning Approach for TinyML On-device Training Using Dataset Distillation and Model Size Adaption**|Marcus Rüb et.al.|[2409.07114](http://arxiv.org/abs/2409.07114)|null|
|**2024-09-11**|**Privacy-Preserving Federated Learning with Consistency via Knowledge Distillation Using Conditional Generator**|Kangyang Luo et.al.|[2409.06955](http://arxiv.org/abs/2409.06955)|null|
|**2024-09-10**|**Applied Federated Model Personalisation in the Industrial Domain: A Comparative Study**|Ilias Siniosoglou et.al.|[2409.06904](http://arxiv.org/abs/2409.06904)|null|
|**2024-09-10**|**EasyST: A Simple Framework for Spatio-Temporal Prediction**|Jiabin Tang et.al.|[2409.06748](http://arxiv.org/abs/2409.06748)|**[link](https://github.com/hkuds/easyst)**|
|**2024-09-10**|**SaRA: High-Efficient Diffusion Model Fine-tuning with Progressive Sparse Low-Rank Adaptation**|Teng Hu et.al.|[2409.06633](http://arxiv.org/abs/2409.06633)|null|
|**2024-09-10**|**Knowledge Distillation via Query Selection for Detection Transformer**|Yi Liu et.al.|[2409.06443](http://arxiv.org/abs/2409.06443)|null|
|**2024-09-10**|**Distilling Generative-Discriminative Representations for Very Low-Resolution Face Recognition**|Junzheng Zhang et.al.|[2409.06371](http://arxiv.org/abs/2409.06371)|null|
|**2024-09-10**|**Enhancing Long Video Understanding via Hierarchical Event-Based Memory**|Dingxin Cheng et.al.|[2409.06299](http://arxiv.org/abs/2409.06299)|null|
|**2024-09-09**|**Joint Input and Output Coordination for Class-Incremental Learning**|Shuai Wang et.al.|[2409.05620](http://arxiv.org/abs/2409.05620)|null|
|**2024-09-09**|**LEROjD: Lidar Extended Radar-Only Object Detection**|Patrick Palmer et.al.|[2409.05564](http://arxiv.org/abs/2409.05564)|**[link](https://github.com/rst-tu-dortmund/lerojd)**|
|**2024-09-09**|**Federated Transfer Learning Based Cooperative Wideband Spectrum Sensing with Model Pruning**|Jibin Jia et.al.|[2409.05462](http://arxiv.org/abs/2409.05462)|null|
|**2024-09-09**|**Look One and More: Distilling Hybrid Order Relational Knowledge for Cross-Resolution Image Recognition**|Shiming Ge et.al.|[2409.05384](http://arxiv.org/abs/2409.05384)|null|
|**2024-09-09**|**Application Specific Compression of Deep Learning Models**|Rohit Raj Rai et.al.|[2409.05368](http://arxiv.org/abs/2409.05368)|**[link](https://github.com/rohitrai11/application-specific-compression-of-deep-learning-models)**|
|**2024-09-09**|**FedBrain-Distill: Communication-Efficient Federated Brain Tumor Classification Using Ensemble Knowledge Distillation on Non-IID Data**|Rasoul Jafari Gohari et.al.|[2409.05359](http://arxiv.org/abs/2409.05359)|**[link](https://github.com/russelljeffrey/FedBrain-Distill)**|
|**2024-09-08**|**Ultron: Enabling Temporal Geometry Compression of 3D Mesh Sequences using Temporal Correspondence and Mesh Deformation**|Haichao Zhu et.al.|[2409.05151](http://arxiv.org/abs/2409.05151)|null|
|**2024-09-07**|**LoCa: Logit Calibration for Knowledge Distillation**|Runming Yang et.al.|[2409.04778](http://arxiv.org/abs/2409.04778)|null|
|**2024-09-06**|**SCARF: Scalable Continual Learning Framework for Memory-efficient Multiple Neural Radiance Fields**|Yuze Wang et.al.|[2409.04482](http://arxiv.org/abs/2409.04482)|null|
|**2024-09-05**|**Experimentation in Content Moderation using RWKV**|Umut Yildirim et.al.|[2409.03939](http://arxiv.org/abs/2409.03939)|null|
|**2024-09-05**|**DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture**|Qianlong Xiang et.al.|[2409.03550](http://arxiv.org/abs/2409.03550)|null|
|**2024-09-05**|**Data-free Distillation with Degradation-prompt Diffusion for Multi-weather Image Restoration**|Pei Wang et.al.|[2409.03455](http://arxiv.org/abs/2409.03455)|null|
|**2024-09-05**|**Efficient Image Compression Using Advanced State Space Models**|Bouzid Arezki et.al.|[2409.02743](http://arxiv.org/abs/2409.02743)|null|
|**2024-09-04**|**CLDA: Collaborative Learning for Enhanced Unsupervised Domain Adaptation**|Minhee Cho et.al.|[2409.02699](http://arxiv.org/abs/2409.02699)|null|
|**2024-09-04**|**Low-Resolution Object Recognition with Cross-Resolution Relational Contrastive Distillation**|Kangkai Zhang et.al.|[2409.02555](http://arxiv.org/abs/2409.02555)|null|
|**2024-09-04**|**A design of magnetic tunnel junctions for the deployment of neuromorphic hardware for edge computing**|Davi Rodrigues et.al.|[2409.02528](http://arxiv.org/abs/2409.02528)|null|
|**2024-09-04**|**Non-target Divergence Hypothesis: Toward Understanding Domain Gaps in Cross-Modal Knowledge Distillation**|Yilong Chen et.al.|[2409.02438](http://arxiv.org/abs/2409.02438)|null|
|**2024-09-03**|**Low-Resolution Face Recognition via Adaptable Instance-Relation Distillation**|Ruixin Shi et.al.|[2409.02049](http://arxiv.org/abs/2409.02049)|null|
|**2024-09-03**|**Foundations of Large Language Model Compression -- Part 1: Weight Quantization**|Sean I. Young et.al.|[2409.02026](http://arxiv.org/abs/2409.02026)|**[link](https://github.com/seannz/cvxq)**|
|**2024-09-03**|**Efficient Point Cloud Classification via Offline Distillation Framework and Negative-Weight Self-Distillation Technique**|Qiang Zheng et.al.|[2409.02020](http://arxiv.org/abs/2409.02020)|null|
|**2024-09-03**|**Contemporary Model Compression on Large Language Models Inference**|Dong Liu et.al.|[2409.01990](http://arxiv.org/abs/2409.01990)|null|
|**2024-09-03**|**Adaptive Explicit Knowledge Transfer for Knowledge Distillation**|Hyungkeun Park et.al.|[2409.01679](http://arxiv.org/abs/2409.01679)|null|
|**2024-08-30**|**How Knowledge Distillation Mitigates the Synthetic Gap in Fair Face Recognition**|Pedro C. Neto et.al.|[2408.17399](http://arxiv.org/abs/2408.17399)|**[link](https://github.com/ivonacolakovic/synthgap-mitigation-using-kd-in-ffr)**|
|**2024-08-30**|**HiTSR: A Hierarchical Transformer for Reference-based Super-Resolution**|Masoomeh Aslahishahri et.al.|[2408.16959](http://arxiv.org/abs/2408.16959)|**[link](https://github.com/bia006/hitsr)**|
|**2024-08-29**|**VLM-KD: Knowledge Distillation from VLM for Long-Tail Visual Recognition**|Zaiwei Zhang et.al.|[2408.16930](http://arxiv.org/abs/2408.16930)|null|
|**2024-08-29**|**Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling**|Hritik Bansal et.al.|[2408.16737](http://arxiv.org/abs/2408.16737)|null|
|**2024-08-29**|**MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition**|Eduarda Caldeira et.al.|[2408.16563](http://arxiv.org/abs/2408.16563)|**[link](https://github.com/eduardacaldeira/mst-kd)**|
|**2024-08-29**|**Convolutional Neural Network Compression Based on Low-Rank Decomposition**|Yaping He et.al.|[2408.16289](http://arxiv.org/abs/2408.16289)|null|
|**2024-08-28**|**LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation**|Fangxun Shu et.al.|[2408.15881](http://arxiv.org/abs/2408.15881)|**[link](https://github.com/shufangxun/llava-mod)**|
|**2024-08-28**|**ModalityMirror: Improving Audio Classification in Modality Heterogeneity Federated Learning with Multimodal Distillation**|Tiantian Feng et.al.|[2408.15803](http://arxiv.org/abs/2408.15803)|null|
|**2024-08-28**|**Online pre-training with long-form videos**|Itsuki Kato et.al.|[2408.15651](http://arxiv.org/abs/2408.15651)|null|
|**2024-08-28**|**Boosting Lossless Speculative Decoding via Feature Sampling and Partial Alignment Distillation**|Lujun Gui et.al.|[2408.15562](http://arxiv.org/abs/2408.15562)|null|
|**2024-08-27**|**Leveraging Self-supervised Audio Representations for Data-Efficient Acoustic Scene Classification**|Yiqiang Cai et.al.|[2408.14862](http://arxiv.org/abs/2408.14862)|**[link](https://github.com/yqcai888/easy_dcase_task1)**|
|**2024-08-27**|**Learning effective pruning at initialization from iterative pruning**|Shengkai Liu et.al.|[2408.14757](http://arxiv.org/abs/2408.14757)|**[link](https://github.com/ChengYaofeng/AutoSparse)**|
|**2024-08-26**|**Bridging the Gap: Unpacking the Hidden Challenges in Knowledge Distillation for Online Ranking Systems**|Nikhil Khani et.al.|[2408.14678](http://arxiv.org/abs/2408.14678)|null|
|**2024-08-25**|**Variational autoencoder-based neural network model compression**|Liang Cheng et.al.|[2408.14513](http://arxiv.org/abs/2408.14513)|null|
|**2024-08-26**|**TSAK: Two-Stage Semantic-Aware Knowledge Distillation for Efficient Wearable Modality and Model Optimization in Manufacturing Lines**|Hymalai Bello et.al.|[2408.14146](http://arxiv.org/abs/2408.14146)|null|
|**2024-08-27**|**GenFormer -- Generated Images are All You Need to Improve Robustness of Transformers on Small Datasets**|Sven Oehri et.al.|[2408.14131](http://arxiv.org/abs/2408.14131)|**[link](https://github.com/cemos-is/genformer)**|
|**2024-08-26**|**Let Video Teaches You More: Video-to-Image Knowledge Distillation using DEtection TRansformer for Medical Video Lesion Detection**|Yuncheng Jiang et.al.|[2408.14051](http://arxiv.org/abs/2408.14051)|null|
|**2024-08-25**|**Condensed Sample-Guided Model Inversion for Knowledge Distillation**|Kuluhan Binici et.al.|[2408.13850](http://arxiv.org/abs/2408.13850)|null|
|**2024-08-25**|**Bring the Power of Diffusion Model to Defect Detection**|Xuyi Yu et.al.|[2408.13845](http://arxiv.org/abs/2408.13845)|null|
|**2024-08-24**|**Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic**|Yifei He et.al.|[2408.13656](http://arxiv.org/abs/2408.13656)|**[link](https://github.com/yifei-he/localize-and-stitch)**|
|**2024-08-24**|**MPruner: Optimizing Neural Network Size with CKA-Based Mutual Information Pruning**|Seungbeom Hu et.al.|[2408.13482](http://arxiv.org/abs/2408.13482)|null|
|**2024-08-23**|**Growing Deep Neural Network Considering with Similarity between Neurons**|Taigo Sakai et.al.|[2408.13291](http://arxiv.org/abs/2408.13291)|null|
|**2024-08-23**|**Foundational Model for Electron Micrograph Analysis: Instruction-Tuning Small-Scale Language-and-Vision Assistant for Enterprise Adoption**|Sakhinana Sagar Srinivas et.al.|[2408.13248](http://arxiv.org/abs/2408.13248)|null|
|**2024-08-23**|**A Web-Based Solution for Federated Learning with LLM-Based Automation**|Chamith Mawela et.al.|[2408.13010](http://arxiv.org/abs/2408.13010)|null|
|**2024-08-23**|**A Survey on Drowsiness Detection -- Modern Applications and Methods**|Biying Fu et.al.|[2408.12990](http://arxiv.org/abs/2408.12990)|null|
|**2024-08-22**|**Pruning By Explaining Revisited: Optimizing Attribution Methods to Prune CNNs and Transformers**|Sayed Mohammad Vakilzadeh Hatefi et.al.|[2408.12568](http://arxiv.org/abs/2408.12568)|**[link](https://github.com/erfanhatefi/pruning-by-explaining-in-pytorch)**|
|**2024-08-22**|**Interactive DualChecker for Mitigating Hallucinations in Distilling Large Language Models**|Meiyun Wang et.al.|[2408.12326](http://arxiv.org/abs/2408.12326)|**[link](https://github.com/kirawang23/dualchecker)**|
|**2024-08-22**|**Rebalancing Multi-Label Class-Incremental Learning**|Kaile Du et.al.|[2408.12161](http://arxiv.org/abs/2408.12161)|null|
|**2024-08-22**|**Vision-Based Detection of Uncooperative Targets and Components on Small Satellites**|Hannah Grauer et.al.|[2408.12084](http://arxiv.org/abs/2408.12084)|null|
|**2024-08-22**|**Aligning (Medical) LLMs for (Counterfactual) Fairness**|Raphael Poulain et.al.|[2408.12055](http://arxiv.org/abs/2408.12055)|**[link](https://github.com/healthylaife/fairalignmentllm)**|
|**2024-08-22**|**LAKD-Activation Mapping Distillation Based on Local Learning**|Yaoze Zhang et.al.|[2408.11478](http://arxiv.org/abs/2408.11478)|null|
|**2024-08-21**|**A Practical Trigger-Free Backdoor Attack on Neural Networks**|Jiahao Wang et.al.|[2408.11444](http://arxiv.org/abs/2408.11444)|null|
|**2024-08-21**|**Pano2Room: Novel View Synthesis from a Single Indoor Panorama**|Guo Pu et.al.|[2408.11413](http://arxiv.org/abs/2408.11413)|**[link](https://github.com/trickygo/pano2room)**|
|**2024-08-21**|**Domain-invariant Progressive Knowledge Distillation for UAV-based Object Detection**|Liang Yao et.al.|[2408.11407](http://arxiv.org/abs/2408.11407)|null|
|**2024-08-21**|**A Unified Framework for Continual Learning and Machine Unlearning**|Romit Chatterjee et.al.|[2408.11374](http://arxiv.org/abs/2408.11374)|null|
|**2024-08-20**|**SAM-COD: SAM-guided Unified Framework for Weakly-Supervised Camouflaged Object Detection**|Huafeng Chen et.al.|[2408.10760](http://arxiv.org/abs/2408.10760)|null|
|**2024-08-20**|**Generating Synthetic Fair Syntax-agnostic Data by Learning and Distilling Fair Representation**|Md Fahim Sikder et.al.|[2408.10755](http://arxiv.org/abs/2408.10755)|null|
|**2024-08-20**|**Fine-Tuning and Deploying Large Language Models Over Edges: Issues and Approaches**|Yanjie Dong et.al.|[2408.10691](http://arxiv.org/abs/2408.10691)|null|
|**2024-08-20**|**LLM-Barber: Block-Aware Rebuilder for Sparsity Mask in One-Shot for Large Language Models**|Yupeng Su et.al.|[2408.10631](http://arxiv.org/abs/2408.10631)|**[link](https://github.com/yupengsu/llm-barber)**|
|**2024-08-20**|**Adaptive Knowledge Distillation for Classification of Hand Images using Explainable Vision Transformers**|Thanh Thi Nguyen et.al.|[2408.10503](http://arxiv.org/abs/2408.10503)|null|
|**2024-08-19**|**Transferring Backdoors between Large Language Models by Knowledge Distillation**|Pengzhou Cheng et.al.|[2408.09878](http://arxiv.org/abs/2408.09878)|**[link](https://github.com/zhou-cybersecurity-ai/atba)**|
|**2024-08-20**|**MoDeGPT: Modular Decomposition for Large Language Model Compression**|Chi-Heng Lin et.al.|[2408.09632](http://arxiv.org/abs/2408.09632)|null|
|**2024-08-18**|**MedMAP: Promoting Incomplete Multi-modal Brain Tumor Segmentation with Alignment**|Tianyi Liu et.al.|[2408.09465](http://arxiv.org/abs/2408.09465)|null|
|**2024-08-18**|**CLIP-CID: Efficient CLIP Distillation via Cluster-Instance Discrimination**|Kaicheng Yang et.al.|[2408.09441](http://arxiv.org/abs/2408.09441)|null|
|**2024-08-18**|**OVOSE: Open-Vocabulary Semantic Segmentation in Event-Based Cameras**|Muhammad Rameez Ur Rahman et.al.|[2408.09424](http://arxiv.org/abs/2408.09424)|**[link](https://github.com/ram95d/ovose)**|
|**2024-08-17**|**RepControlNet: ControlNet Reparameterization**|Zhaoli Deng et.al.|[2408.09240](http://arxiv.org/abs/2408.09240)|null|
|**2024-08-16**|**Multi Teacher Privileged Knowledge Distillation for Multimodal Expression Recognition**|Muhammad Haseeb Aslam et.al.|[2408.09035](http://arxiv.org/abs/2408.09035)|**[link](https://github.com/haseebaslam95/MT-PKDOT)**|
|**2024-08-16**|**Research on Personalized Compression Algorithm for Pre-trained Models Based on Homomorphic Entropy Increase**|Yicong Li et.al.|[2408.08684](http://arxiv.org/abs/2408.08684)|null|
|**2024-08-16**|**ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration for Large Language Models**|Chao Zeng et.al.|[2408.08554](http://arxiv.org/abs/2408.08554)|**[link](https://github.com/bytedance/abq-llm)**|
|**2024-08-15**|**Computer Vision Model Compression Techniques for Embedded Systems: A Survey**|Alexandre Lopes et.al.|[2408.08250](http://arxiv.org/abs/2408.08250)|**[link](https://github.com/venturusbr/cv-model-compression)**|
|**2024-08-15**|**MIDAS: Multi-level Intent, Domain, And Slot Knowledge Distillation for Multi-turn NLU**|Yan Li et.al.|[2408.08144](http://arxiv.org/abs/2408.08144)|null|
|**2024-08-19**|**Knowledge Distillation with Refined Logits**|Wujie Sun et.al.|[2408.07703](http://arxiv.org/abs/2408.07703)|**[link](https://github.com/zju-swj/rld)**|
|**2024-08-14**|**FedQUIT: On-Device Federated Unlearning via a Quasi-Competent Virtual Teacher**|Alessio Mora et.al.|[2408.07587](http://arxiv.org/abs/2408.07587)|null|
|**2024-08-14**|**Towards Real-time Video Compressive Sensing on Mobile Devices**|Miao Cao et.al.|[2408.07530](http://arxiv.org/abs/2408.07530)|**[link](https://github.com/mcao92/mobilesci)**|
|**2024-08-14**|**One Step Diffusion-based Super-Resolution with Time-Aware Distillation**|Xiao He et.al.|[2408.07476](http://arxiv.org/abs/2408.07476)|**[link](https://github.com/learninghx/tad-sr)**|
|**2024-08-14**|**Infra-YOLO: Efficient Neural Network Structure with Model Compression for Real-Time Infrared Small Object Detection**|Zhonglin Chen et.al.|[2408.07455](http://arxiv.org/abs/2408.07455)|null|
|**2024-08-13**|**Using Advanced LLMs to Enhance Smaller LLMs: An Interpretable Knowledge Distillation Approach**|Tong Wang et.al.|[2408.07238](http://arxiv.org/abs/2408.07238)|null|
|**2024-08-15**|**An Event Structure-aware Generative Model for Biomedical Event Extraction**|Haohan Yuan et.al.|[2408.06583](http://arxiv.org/abs/2408.06583)|null|
|**2024-08-12**|**Optimizing Vision Transformers with Data-Free Knowledge Transfer**|Gousia Habib et.al.|[2408.05952](http://arxiv.org/abs/2408.05952)|null|
|**2024-08-11**|**Low-Dimensional Federated Knowledge Graph Embedding via Knowledge Distillation**|Xiaoxiong Zhang et.al.|[2408.05748](http://arxiv.org/abs/2408.05748)|null|
|**2024-08-11**|**Efficient Federated Learning Using Dynamic Update and Adaptive Pruning with Momentum on Shared Server Data**|Ji Liu et.al.|[2408.05678](http://arxiv.org/abs/2408.05678)|null|
|**2024-08-08**|**LaDiMo: Layer-wise Distillation Inspired MoEfier**|Sungyoon Kim et.al.|[2408.04278](http://arxiv.org/abs/2408.04278)|null|
|**2024-08-08**|**Distil-DCCRN: A Small-footprint DCCRN Leveraging Feature-based Knowledge Distillation in Speech Enhancement**|Runduo Han et.al.|[2408.04267](http://arxiv.org/abs/2408.04267)|null|
|**2024-08-14**|**ComKD-CLIP: Comprehensive Knowledge Distillation for Contrastive Language-Image Pre-traning Model**|Yifan Chen et.al.|[2408.04145](http://arxiv.org/abs/2408.04145)|null|
|**2024-08-07**|**AdapMTL: Adaptive Pruning Framework for Multitask Learning Model**|Mingcan Xiang et.al.|[2408.03913](http://arxiv.org/abs/2408.03913)|null|
|**2024-08-07**|**Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection**|Xinyue Liu et.al.|[2408.03888](http://arxiv.org/abs/2408.03888)|null|
|**2024-08-07**|**Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields**|Joo Chan Lee et.al.|[2408.03822](http://arxiv.org/abs/2408.03822)|null|
|**2024-08-07**|**Iterative Knowledge Distillation through Feedback-Driven Learning Cycles**|Yujia Chen et.al.|[2408.03680](http://arxiv.org/abs/2408.03680)|null|
|**2024-08-07**|**Real-time Event Recognition of Long-distance Distributed Vibration Sensing with Knowledge Distillation and Hardware Acceleration**|Zhongyao Luo et.al.|[2408.03647](http://arxiv.org/abs/2408.03647)|**[link](https://github.com/hust-iof/efficient-dvs)**|
|**2024-08-07**|**Distillation Learning Guided by Image Reconstruction for One-Shot Medical Image Segmentation**|Feng Zhou et.al.|[2408.03616](http://arxiv.org/abs/2408.03616)|**[link](https://github.com/novicefodder/os-medseg)**|
|**2024-08-06**|**EEGMobile: Enhancing Speed and Accuracy in EEG-Based Gaze Prediction with Advanced Mobile Architectures**|Teng Liang et.al.|[2408.03449](http://arxiv.org/abs/2408.03449)|**[link](https://github.com/t0nyliang/EEGMobile)**|
|**2024-08-06**|**DopQ-ViT: Towards Distribution-Friendly and Outlier-Aware Post-Training Quantization for Vision Transformers**|Lianwei Yang et.al.|[2408.03291](http://arxiv.org/abs/2408.03291)|null|
|**2024-08-06**|**Compress and Compare: Interactively Evaluating Efficiency and Behavior Across ML Model Compression Experiments**|Angie Boggust et.al.|[2408.03274](http://arxiv.org/abs/2408.03274)|null|
|**2024-08-06**|**Leveraging Entity Information for Cross-Modality Correlation Learning: The Entity-Guided Multimodal Summarization**|Yanghai Zhang et.al.|[2408.03149](http://arxiv.org/abs/2408.03149)|**[link](https://github.com/ApocalypseH/EGMS)**|
|**2024-08-06**|**Inference Optimizations for Large Language Models: Effects, Challenges, and Practical Considerations**|Leo Donisch et.al.|[2408.03130](http://arxiv.org/abs/2408.03130)|null|
|**2024-08-06**|**Comb, Prune, Distill: Towards Unified Pruning for Vision Model Compression**|Jonas Schmitt et.al.|[2408.03046](http://arxiv.org/abs/2408.03046)|**[link](https://github.com/cranken/cpd)**|
|**2024-08-06**|**VizECGNet: Visual ECG Image Network for Cardiovascular Diseases Classification with Multi-Modal Training and Knowledge Distillation**|Ju-Hyeon Nam et.al.|[2408.02888](http://arxiv.org/abs/2408.02888)|null|
|**2024-08-05**|**An approach to optimize inference of the DIART speaker diarization pipeline**|Roman Aperdannier et.al.|[2408.02341](http://arxiv.org/abs/2408.02341)|null|
|**2024-08-05**|**Low-Cost Self-Ensembles Based on Multi-Branch Transformation and Grouped Convolution**|Hojung Lee et.al.|[2408.02307](http://arxiv.org/abs/2408.02307)|**[link](https://github.com/hjdw2/sembg)**|
|**2024-08-05**|**Unsupervised Domain Adaption Harnessing Vision-Language Pre-training**|Wenlve Zhou et.al.|[2408.02192](http://arxiv.org/abs/2408.02192)|**[link](https://github.com/Wenlve-Zhou/VLP-UDA)**|
|**2024-08-03**|**Joint Model Pruning and Resource Allocation for Wireless Time-triggered Federated Learning**|Xinlu Zhang et.al.|[2408.01765](http://arxiv.org/abs/2408.01765)|null|
|**2024-08-02**|**An Adaptive Tensor-Train Decomposition Approach for Efficient Deep Neural Network Compression**|Shiyi Luo et.al.|[2408.01534](http://arxiv.org/abs/2408.01534)|null|
|**2024-08-02**|**Exploiting the Semantic Knowledge of Pre-trained Text-Encoders for Continual Learning**|Lu Yu et.al.|[2408.01076](http://arxiv.org/abs/2408.01076)|**[link](https://github.com/aprilsveryown/semantically-guided-continual-learning)**|
|**2024-08-02**|**Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs**|Afia Anjum et.al.|[2408.01008](http://arxiv.org/abs/2408.01008)|null|
|**2024-08-01**|**DistillGrasp: Integrating Features Correlation with Knowledge Distillation for Depth Completion of Transparent Objects**|Yiheng Huang et.al.|[2408.00337](http://arxiv.org/abs/2408.00337)|null|
|**2024-08-01**|**Clover-2: Accurate Inference for Regressive Lightweight Speculative Decoding**|Bin Xiao et.al.|[2408.00264](http://arxiv.org/abs/2408.00264)|null|
|**2024-08-01**|**Sentence-wise Speech Summarization: Task, Datasets, and End-to-End Modeling with LM Knowledge Distillation**|Kohei Matsuura et.al.|[2408.00205](http://arxiv.org/abs/2408.00205)|null|
|**2024-07-31**|**StyleRF-VolVis: Style Transfer of Neural Radiance Fields for Expressive Volume Visualization**|Kaiyuan Tang et.al.|[2408.00150](http://arxiv.org/abs/2408.00150)|null|
|**2024-08-02**|**Gemma 2: Improving Open Language Models at a Practical Size**|Gemma Team et.al.|[2408.00118](http://arxiv.org/abs/2408.00118)|null|
|**2024-07-31**|**Dynamic Object Queries for Transformer-based Incremental Object Detection**|Jichuan Zhang et.al.|[2407.21687](http://arxiv.org/abs/2407.21687)|null|
|**2024-07-31**|**Learning Effective Representations for Retrieval Using Self-Distillation with Adaptive Relevance Margins**|Lukas Gienapp et.al.|[2407.21515](http://arxiv.org/abs/2407.21515)|null|
|**2024-07-31**|**VIPeR: Visual Incremental Place Recognition with Adaptive Mining and Lifelong Learning**|Yuhang Ming et.al.|[2407.21416](http://arxiv.org/abs/2407.21416)|null|
|**2024-07-31**|**Lifelong Person Search**|Jae-Won Yang et.al.|[2407.21252](http://arxiv.org/abs/2407.21252)|null|
|**2024-07-29**|**SalNAS: Efficient Saliency-prediction Neural Architecture Search with self-knowledge distillation**|Chakkrit Termritthikun et.al.|[2407.20062](http://arxiv.org/abs/2407.20062)|**[link](https://github.com/chakkritte/SalNAS)**|
|**2024-07-29**|**ActivityCLIP: Enhancing Group Activity Recognition by Mining Complementary Information from Text to Supplement Image Modality**|Guoliang Xu et.al.|[2407.19820](http://arxiv.org/abs/2407.19820)|null|
|**2024-07-29**|**Realizing Unaligned Block-wise Pruning for DNN Acceleration on Mobile Devices**|Hayun Lee et.al.|[2407.19644](http://arxiv.org/abs/2407.19644)|null|
|**2024-07-28**|**Mixture of Modular Experts: Distilling Knowledge from a Multilingual Teacher into Specialized Modular Language Models**|Mohammed Al-Maamari et.al.|[2407.19610](http://arxiv.org/abs/2407.19610)|**[link](https://github.com/padas-lab-de/multi-language-dataset-creator)**|
|**2024-07-28**|**Overcoming Uncertain Incompleteness for Robust Multimodal Sequential Diagnosis Prediction via Knowledge Distillation and Random Data Erasing**|Heejoon Koo et.al.|[2407.19540](http://arxiv.org/abs/2407.19540)|null|
|**2024-07-28**|**LLAVADI: What Matters For Multimodal Large Language Models Distillation**|Shilin Xu et.al.|[2407.19409](http://arxiv.org/abs/2407.19409)|null|
|**2024-07-28**|**Logic Distillation: Learning from Code Function by Function for Planning and Decision-making**|Dong Chen et.al.|[2407.19405](http://arxiv.org/abs/2407.19405)|null|
|**2024-07-27**|**Sewer Image Super-Resolution with Depth Priors and Its Lightweight Network**|Gang Pan et.al.|[2407.19271](http://arxiv.org/abs/2407.19271)|null|
|**2024-07-26**|**Automatic Detection of Moral Values in Music Lyrics**|Vjosa Preniqi et.al.|[2407.18787](http://arxiv.org/abs/2407.18787)|**[link](https://github.com/vjosapreniqi/ismir-mft-values)**|
|**2024-07-26**|**Boosting Cross-Domain Point Classification via Distilling Relational Priors from 2D Transformers**|Longkun Zou et.al.|[2407.18534](http://arxiv.org/abs/2407.18534)|**[link](https://github.com/zou-longkun/rpd)**|
|**2024-07-26**|**FedUD: Exploiting Unaligned Data for Cross-Platform Federated Click-Through Rate Prediction**|Wentao Ouyang et.al.|[2407.18472](http://arxiv.org/abs/2407.18472)|null|
|**2024-07-26**|**Towards A Generalizable Pathology Foundation Model via Unified Knowledge Distillation**|Jiabo Ma et.al.|[2407.18449](http://arxiv.org/abs/2407.18449)|null|
|**2024-07-25**|**Leveraging Foundation Models via Knowledge Distillation in Multi-Object Tracking: Distilling DINOv2 Features to FairMOT**|Niels G. Faber et.al.|[2407.18288](http://arxiv.org/abs/2407.18288)|**[link](https://github.com/NissaFaber/Thesis_repo)**|
|**2024-07-25**|**Self-Training with Direct Preference Optimization Improves Chain-of-Thought Reasoning**|Tianduo Wang et.al.|[2407.18248](http://arxiv.org/abs/2407.18248)|**[link](https://github.com/tianduowang/dpo-st)**|
|**2024-07-25**|**How to Train the Teacher Model for Effective Knowledge Distillation**|Shayan Mohajer Hamidi et.al.|[2407.18041](http://arxiv.org/abs/2407.18041)|**[link](https://github.com/eccv2024mse/eccv_mse_teacher)**|
|**2024-07-25**|**Peak-Controlled Logits Poisoning Attack in Federated Distillation**|Yuhan Tang et.al.|[2407.18039](http://arxiv.org/abs/2407.18039)|null|
|**2024-07-25**|**Separating Novel Features for Logical Anomaly Detection: A Straightforward yet Effective Approach**|Kangil Lee et.al.|[2407.17909](http://arxiv.org/abs/2407.17909)|null|
|**2024-07-25**|**NC-NCD: Novel Class Discovery for Node Classification**|Yue Hou et.al.|[2407.17816](http://arxiv.org/abs/2407.17816)|**[link](https://github.com/name-is-what/nc2d)**|
|**2024-07-24**|**CoMoTo: Unpaired Cross-Modal Lesion Distillation Improves Breast Lesion Detection in Tomosynthesis**|Muhammad Alberb et.al.|[2407.17620](http://arxiv.org/abs/2407.17620)|**[link](https://github.com/muhammad-al-barbary/comoto)**|
|**2024-07-24**|**(PASS) Visual Prompt Locates Good Structure Sparsity through a Recurrent HyperNetwork**|Tianjin Huang et.al.|[2407.17412](http://arxiv.org/abs/2407.17412)|null|
|**2024-07-23**|**Strike a Balance in Continual Panoptic Segmentation**|Jinpeng Chen et.al.|[2407.16354](http://arxiv.org/abs/2407.16354)|**[link](https://github.com/jinpeng0528/balconpas)**|
|**2024-07-23**|**OriGen:Enhancing RTL Code Generation with Code-to-Code Augmentation and Self-Reflection**|Fan Cui et.al.|[2407.16237](http://arxiv.org/abs/2407.16237)|**[link](https://github.com/pku-liang/origen)**|
|**2024-07-23**|**DDK: Distilling Domain Knowledge for Efficient Large Language Models**|Jiaheng Liu et.al.|[2407.16154](http://arxiv.org/abs/2407.16154)|null|

<p align=right>(<a href=#updated-on-20241231>back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[contributors-url]: https://github.com/Vincentqyw/cv-arxiv-daily/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[forks-url]: https://github.com/Vincentqyw/cv-arxiv-daily/network/members
[stars-shield]: https://img.shields.io/github/stars/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[stars-url]: https://github.com/Vincentqyw/cv-arxiv-daily/stargazers
[issues-shield]: https://img.shields.io/github/issues/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[issues-url]: https://github.com/Vincentqyw/cv-arxiv-daily/issues

