[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

## Updated on 2024.09.02
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
|**2024-08-29**|**A machine learning approach for computing solar flare locations in X-rays on-board Solar Orbiter/STIX**|Paolo Massa et.al.|[2408.16642](http://arxiv.org/abs/2408.16642)|null|
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

<p align=right>(<a href=#updated-on-20240902>back to top</a>)</p>

## Pruning

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
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

<p align=right>(<a href=#updated-on-20240902>back to top</a>)</p>

## Hardware-Software Co-Design

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-08-29**|**On-device AI: Quantization-aware Training of Transformers in Time-Series**|Tianheng Ling et.al.|[2408.16495](http://arxiv.org/abs/2408.16495)|null|
|**2024-08-29**|**Accelerating Image-based Pest Detection on a Heterogeneous Multi-core Microcontroller**|Luca Bompani et.al.|[2408.15911](http://arxiv.org/abs/2408.15911)|null|
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

<p align=right>(<a href=#updated-on-20240902>back to top</a>)</p>

## TinyML

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-08-29**|**TinyTNAS: GPU-Free, Time-Bound, Hardware-Aware Neural Architecture Search for TinyML Time Series Classification**|Bidyut Saha et.al.|[2408.16535](http://arxiv.org/abs/2408.16535)|null|
|**2024-08-08**|**An Edge AI System Based on FPGA Platform for Railway Fault Detection**|Jiale Li et.al.|[2408.15245](http://arxiv.org/abs/2408.15245)|null|
|**2024-08-23**|**S3Simulator: A benchmarking Side Scan Sonar Simulator dataset for Underwater Image Analysis**|Kamal Basha S et.al.|[2408.12833](http://arxiv.org/abs/2408.12833)|null|
|**2024-08-20**|**Pluto and Charon: A Time and Memory Efficient Collaborative Edge AI Framework for Personal LLMs Fine-Tuning**|Bei Ouyang et.al.|[2408.10746](http://arxiv.org/abs/2408.10746)|null|
|**2024-08-21**|**Challenges and Responses in the Practice of Large Language Models**|Hongyin Zhu et.al.|[2408.09416](http://arxiv.org/abs/2408.09416)|null|
|**2024-08-15**|**Moving Healthcare AI-Support Systems for Visually Detectable Diseases onto Constrained Devices**|Tess Watt et.al.|[2408.08215](http://arxiv.org/abs/2408.08215)|null|
|**2024-08-14**|**Efficient Edge AI: Deploying Convolutional Neural Networks on FPGA with the Gemmini Accelerator**|Federico Nicolas Peccia et.al.|[2408.07404](http://arxiv.org/abs/2408.07404)|null|
|**2024-08-13**|**Harnessing Earnings Reports for Stock Predictions: A QLoRA-Enhanced LLM Approach**|Haowei Ni et.al.|[2408.06634](http://arxiv.org/abs/2408.06634)|null|
|**2024-08-06**|**Training on the Fly: On-device Self-supervised Learning aboard Nano-drones within 20 mW**|Elia Cereda et.al.|[2408.03168](http://arxiv.org/abs/2408.03168)|null|
|**2024-08-05**|**Toward Attention-based TinyML: A Heterogeneous Accelerated Architecture and Automated Deployment Flow**|Philip Wiese et.al.|[2408.02473](http://arxiv.org/abs/2408.02473)|null|
|**2024-08-05**|**PENDRAM: Enabling High-Performance and Energy-Efficient Processing of Deep Neural Networks through a Generalized DRAM Data Mapping Policy**|Rachmad Vidya Wicaksana Putra et.al.|[2408.02412](http://arxiv.org/abs/2408.02412)|null|
|**2024-08-02**|**A Tiny Supervised ODL Core with Auto Data Pruning for Human Activity Recognition**|Hiroki Matsutani et.al.|[2408.01283](http://arxiv.org/abs/2408.01283)|null|
|**2024-07-29**|**HOAA: Hybrid Overestimating Approximate Adder for Enhanced Performance Processing Engine**|Omkar Kokane et.al.|[2408.00806](http://arxiv.org/abs/2408.00806)|null|
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
|**2024-07-16**|**XEdgeAI: A Human-centered Industrial Inspection Framework with Data-centric Explainable Edge AI Approach**|Truong Thanh Hung Nguyen et.al.|[2407.11771](http://arxiv.org/abs/2407.11771)|null|
|**2024-07-18**|**Enhancing TinyML Security: Study of Adversarial Attack Transferability**|Parin Shah et.al.|[2407.11599](http://arxiv.org/abs/2407.11599)|null|
|**2024-07-13**|**Characterizing Disparity Between Edge Models and High-Accuracy Base Models for Vision Tasks**|Zhenyu Wang et.al.|[2407.10016](http://arxiv.org/abs/2407.10016)|null|
|**2024-07-11**|**Towards Efficient Deployment of Hybrid SNNs on Neuromorphic and Edge AI Hardware**|James Seekings et.al.|[2407.08704](http://arxiv.org/abs/2407.08704)|null|

<p align=right>(<a href=#updated-on-20240902>back to top</a>)</p>

## Domain Specific Accelerator

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-08-27**|**SiHGNN: Leveraging Properties of Semantic Graphs for Efficient HGNN Acceleration**|Runzhen Xue et.al.|[2408.15089](http://arxiv.org/abs/2408.15089)|null|
|**2024-08-24**|**SiTe CiM: Signed Ternary Computing-in-Memory for Ultra-Low Precision Deep Neural Networks**|Niharika Thakuria et.al.|[2408.13617](http://arxiv.org/abs/2408.13617)|null|
|**2024-08-13**|**Potamoi: Accelerating Neural Rendering via a Unified Streaming Architecture**|Yu Feng et.al.|[2408.06608](http://arxiv.org/abs/2408.06608)|null|
|**2024-08-09**|**Scaling Deep Learning Computation over the Inter-Core Connected Intelligence Processor**|Yiqi Liu et.al.|[2408.04808](http://arxiv.org/abs/2408.04808)|null|
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

<p align=right>(<a href=#updated-on-20240902>back to top</a>)</p>

## Low-Rank Adaptation

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-08-29**|**LoraMap: Harnessing the Power of LoRA Connections**|Hyeryun Park et.al.|[2408.16264](http://arxiv.org/abs/2408.16264)|null|
|**2024-08-28**|**LeMON: Learning to Learn Multi-Operator Networks**|Jingmin Sun et.al.|[2408.16168](http://arxiv.org/abs/2408.16168)|**[link](https://github.com/jingminsun/lemon_prose)**|
|**2024-08-28**|**Leveraging Open Knowledge for Advancing Task Expertise in Large Language Models**|Yuncheng Yang et.al.|[2408.15915](http://arxiv.org/abs/2408.15915)|null|
|**2024-08-28**|**StyleRemix: Interpretable Authorship Obfuscation via Distillation and Perturbation of Style Elements**|Jillian Fisher et.al.|[2408.15666](http://arxiv.org/abs/2408.15666)|**[link](https://github.com/jfisher52/StyleRemix)**|
|**2024-08-28**|**TeFF: Tracking-enhanced Forgetting-free Few-shot 3D LiDAR Semantic Segmentation**|Junbao Zhou et.al.|[2408.15657](http://arxiv.org/abs/2408.15657)|**[link](https://github.com/junbao-zhou/track-no-forgetting)**|
|**2024-08-28**|**Whisper-PMFA: Partial Multi-Scale Feature Aggregation for Speaker Verification using Whisper Models**|Yiyang Zhao et.al.|[2408.15585](http://arxiv.org/abs/2408.15585)|null|
|**2024-08-28**|**VoiceTailor: Lightweight Plug-In Adapter for Diffusion-Based Personalized Text-to-Speech**|Heeseung Kim et.al.|[2408.14739](http://arxiv.org/abs/2408.14739)|null|
|**2024-08-27**|**PAT: Pruning-Aware Tuning for Large Language Models**|Yijiang Liu et.al.|[2408.14721](http://arxiv.org/abs/2408.14721)|**[link](https://github.com/kriskrisliu/pat_pruning-aware-tuning)**|
|**2024-08-27**|**StyleSpeech: Parameter-efficient Fine Tuning for Pre-trained Controllable Text-to-Speech**|Haowei Lou et.al.|[2408.14713](http://arxiv.org/abs/2408.14713)|null|
|**2024-08-26**|**CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation**|Muhammad Fawi et.al.|[2408.14572](http://arxiv.org/abs/2408.14572)|**[link](https://github.com/mnoorfawi/curlora)**|
|**2024-08-27**|**Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models**|Aradhye Agarwal et.al.|[2408.14470](http://arxiv.org/abs/2408.14470)|**[link](https://github.com/Aradhye2002/selective-peft-toolkit)**|
|**2024-08-26**|**Reprogramming Foundational Large Language Models(LLMs) for Enterprise Adoption for Spatio-Temporal Forecasting Applications: Unveiling a New Era in Copilot-Guided Cross-Modal Time Series Representation Learning**|Sakhinana Sagar Srinivas et.al.|[2408.14387](http://arxiv.org/abs/2408.14387)|null|
|**2024-08-27**|**SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher**|Trung Dao et.al.|[2408.14176](http://arxiv.org/abs/2408.14176)|null|
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
|**2024-08-19**|**Customizing Language Models with Instance-wise LoRA for Sequential Recommendation**|Xiaoyu Kong et.al.|[2408.10159](http://arxiv.org/abs/2408.10159)|null|
|**2024-08-19**|**TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition**|Tianwei Lin et.al.|[2408.09856](http://arxiv.org/abs/2408.09856)|**[link](https://github.com/lin-tianwei/teamlora)**|
|**2024-08-18**|**Infinite Scrolling, Finite Satisfaction: Exploring User Behavior and Satisfaction on Social Media in Bangladesh**|Sanzana Karim Lora et.al.|[2408.09601](http://arxiv.org/abs/2408.09601)|null|
|**2024-08-17**|**ConVerSum: A Contrastive Learning based Approach for Data-Scarce Solution of Cross-Lingual Summarization Beyond Direct Equivalents**|Sanzana Karim Lora et.al.|[2408.09273](http://arxiv.org/abs/2408.09273)|null|
|**2024-08-17**|**An Exploratory Study on Fine-Tuning Large Language Models for Secure Code Generation**|Junjie Li et.al.|[2408.09078](http://arxiv.org/abs/2408.09078)|null|
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
|**2024-08-09**|**TaSL: Task Skill Localization and Consolidation for Language Model Continual Learning**|Yujie Feng et.al.|[2408.05200](http://arxiv.org/abs/2408.05200)|null|
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
|**2024-07-23**|**Harmonizing Visual Text Comprehension and Generation**|Zhen Zhao et.al.|[2407.16364](http://arxiv.org/abs/2407.16364)|null|
|**2024-07-23**|**FoRA: Low-Rank Adaptation Model beyond Multimodal Siamese Network**|Weiying Xie et.al.|[2407.16129](http://arxiv.org/abs/2407.16129)|**[link](https://github.com/zyszxhy/fora)**|
|**2024-07-22**|**Test-Time Low Rank Adaptation via Confidence Maximization for Zero-Shot Generalization of Vision-Language Models**|Raza Imam et.al.|[2407.15913](http://arxiv.org/abs/2407.15913)|**[link](https://github.com/razaimam45/ttl-test-time-low-rank-adaptation)**|
|**2024-07-22**|**Zero-Shot Embeddings Inform Learning and Forgetting with Vision-Language Encoders**|Laura Niss et.al.|[2407.15731](http://arxiv.org/abs/2407.15731)|null|
|**2024-07-22**|**LLaST: Improved End-to-end Speech Translation System Leveraged by Large Language Models**|Xi Chen et.al.|[2407.15415](http://arxiv.org/abs/2407.15415)|**[link](https://github.com/openaudiolab/llast)**|
|**2024-07-21**|**Learn to Preserve and Diversify: Parameter-Efficient Group with Orthogonal Regularization for Domain Generalization**|Jiajun Hu et.al.|[2407.15085](http://arxiv.org/abs/2407.15085)|null|
|**2024-07-21**|**MedSAGa: Few-shot Memory Efficient Medical Image Segmentation using Gradient Low-Rank Projection in SAM**|Navyansh Mahla et.al.|[2407.15042](http://arxiv.org/abs/2407.15042)|null|

<p align=right>(<a href=#updated-on-20240902>back to top</a>)</p>

## Model Compression

|Publish Date|Title|Authors|PDF|Code|
|---|---|---|---|---|
|**2024-08-29**|**Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling**|Hritik Bansal et.al.|[2408.16737](http://arxiv.org/abs/2408.16737)|null|
|**2024-08-29**|**MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition**|Eduarda Caldeira et.al.|[2408.16563](http://arxiv.org/abs/2408.16563)|**[link](https://github.com/eduardacaldeira/mst-kd)**|
|**2024-08-29**|**Convolutional Neural Network Compression Based on Low-Rank Decomposition**|Yaping He et.al.|[2408.16289](http://arxiv.org/abs/2408.16289)|null|
|**2024-08-28**|**LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation**|Fangxun Shu et.al.|[2408.15881](http://arxiv.org/abs/2408.15881)|**[link](https://github.com/shufangxun/llava-mod)**|
|**2024-08-28**|**ModalityMirror: Improving Audio Classification in Modality Heterogeneity Federated Learning with Multimodal Distillation**|Tiantian Feng et.al.|[2408.15803](http://arxiv.org/abs/2408.15803)|null|
|**2024-08-28**|**Online pre-training with long-form videos**|Itsuki Kato et.al.|[2408.15651](http://arxiv.org/abs/2408.15651)|null|
|**2024-08-28**|**Boosting Lossless Speculative Decoding via Feature Sampling and Partial Alignment Distillation**|Lujun Gui et.al.|[2408.15562](http://arxiv.org/abs/2408.15562)|null|
|**2024-08-27**|**Leveraging Self-supervised Audio Representations for Data-Efficient Acoustic Scene Classification**|Yiqiang Cai et.al.|[2408.14862](http://arxiv.org/abs/2408.14862)|null|
|**2024-08-27**|**Learning effective pruning at initialization from iterative pruning**|Shengkai Liu et.al.|[2408.14757](http://arxiv.org/abs/2408.14757)|null|
|**2024-08-26**|**Bridging the Gap: Unpacking the Hidden Challenges in Knowledge Distillation for Online Ranking Systems**|Nikhil Khani et.al.|[2408.14678](http://arxiv.org/abs/2408.14678)|null|
|**2024-08-25**|**Variational autoencoder-based neural network model compression**|Liang Cheng et.al.|[2408.14513](http://arxiv.org/abs/2408.14513)|null|
|**2024-08-26**|**TSAK: Two-Stage Semantic-Aware Knowledge Distillation for Efficient Wearable Modality and Model Optimization in Manufacturing Lines**|Hymalai Bello et.al.|[2408.14146](http://arxiv.org/abs/2408.14146)|null|
|**2024-08-27**|**GenFormer -- Generated Images are All You Need to Improve Robustness of Transformers on Small Datasets**|Sven Oehri et.al.|[2408.14131](http://arxiv.org/abs/2408.14131)|null|
|**2024-08-26**|**Let Video Teaches You More: Video-to-Image Knowledge Distillation using DEtection TRansformer for Medical Video Lesion Detection**|Yuncheng Jiang et.al.|[2408.14051](http://arxiv.org/abs/2408.14051)|null|
|**2024-08-25**|**Condensed Sample-Guided Model Inversion for Knowledge Distillation**|Kuluhan Binici et.al.|[2408.13850](http://arxiv.org/abs/2408.13850)|null|
|**2024-08-25**|**Bring the Power of Diffusion Model to Defect Detection**|Xuyi Yu et.al.|[2408.13845](http://arxiv.org/abs/2408.13845)|null|
|**2024-08-24**|**Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic**|Yifei He et.al.|[2408.13656](http://arxiv.org/abs/2408.13656)|**[link](https://github.com/yifei-he/localize-and-stitch)**|
|**2024-08-24**|**MPruner: Optimizing Neural Network Size with CKA-Based Mutual Information Pruning**|Seungbeom Hu et.al.|[2408.13482](http://arxiv.org/abs/2408.13482)|null|
|**2024-08-23**|**Growing Deep Neural Network Considering with Similarity between Neurons**|Taigo Sakai et.al.|[2408.13291](http://arxiv.org/abs/2408.13291)|null|
|**2024-08-23**|**Foundational Model for Electron Micrograph Analysis: Instruction-Tuning Small-Scale Language-and-Vision Assistant for Enterprise Adoption**|Sakhinana Sagar Srinivas et.al.|[2408.13248](http://arxiv.org/abs/2408.13248)|null|
|**2024-08-23**|**A Web-Based Solution for Federated Learning with LLM-Based Automation**|Chamith Mawela et.al.|[2408.13010](http://arxiv.org/abs/2408.13010)|null|
|**2024-08-23**|**A Survey on Drowsiness Detection -- Modern Applications and Methods**|Biying Fu et.al.|[2408.12990](http://arxiv.org/abs/2408.12990)|null|
|**2024-08-22**|**Pruning By Explaining Revisited: Optimizing Attribution Methods to Prune CNNs and Transformers**|Sayed Mohammad Vakilzadeh Hatefi et.al.|[2408.12568](http://arxiv.org/abs/2408.12568)|null|
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
|**2024-08-16**|**ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration for Large Language Models**|Chao Zeng et.al.|[2408.08554](http://arxiv.org/abs/2408.08554)|null|
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
|**2024-07-25**|**NC-NCD: Novel Class Discovery for Node Classification**|Yue Hou et.al.|[2407.17816](http://arxiv.org/abs/2407.17816)|null|
|**2024-07-24**|**CoMoTo: Unpaired Cross-Modal Lesion Distillation Improves Breast Lesion Detection in Tomosynthesis**|Muhammad Alberb et.al.|[2407.17620](http://arxiv.org/abs/2407.17620)|**[link](https://github.com/muhammad-al-barbary/comoto)**|
|**2024-07-24**|**(PASS) Visual Prompt Locates Good Structure Sparsity through a Recurrent HyperNetwork**|Tianjin Huang et.al.|[2407.17412](http://arxiv.org/abs/2407.17412)|null|
|**2024-07-23**|**Strike a Balance in Continual Panoptic Segmentation**|Jinpeng Chen et.al.|[2407.16354](http://arxiv.org/abs/2407.16354)|**[link](https://github.com/jinpeng0528/balconpas)**|
|**2024-07-23**|**OriGen:Enhancing RTL Code Generation with Code-to-Code Augmentation and Self-Reflection**|Fan Cui et.al.|[2407.16237](http://arxiv.org/abs/2407.16237)|null|
|**2024-07-23**|**DDK: Distilling Domain Knowledge for Efficient Large Language Models**|Jiaheng Liu et.al.|[2407.16154](http://arxiv.org/abs/2407.16154)|null|

<p align=right>(<a href=#updated-on-20240902>back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[contributors-url]: https://github.com/Vincentqyw/cv-arxiv-daily/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[forks-url]: https://github.com/Vincentqyw/cv-arxiv-daily/network/members
[stars-shield]: https://img.shields.io/github/stars/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[stars-url]: https://github.com/Vincentqyw/cv-arxiv-daily/stargazers
[issues-shield]: https://img.shields.io/github/issues/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge
[issues-url]: https://github.com/Vincentqyw/cv-arxiv-daily/issues

