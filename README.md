# CoGenesis

## Abstract
With the advancement of language models (LMs), their exposure to private data is increasingly inevitable, and their deployment (especially for smaller ones) on personal devices, such as PCs and smartphones, has become a prevailing trend. In contexts laden with user information, enabling models to both safeguard user privacy and execute commands efficiently emerges as an essential research imperative. In this paper, we propose CoGenesis, a collaborative generation framework integrating large (hosted on cloud infrastructure) and small models (deployed on local devices) to address privacy concerns logically. Initially, we design a pipeline to create personalized writing instruction datasets enriched with extensive context details as the testbed of this research issue. Subsequently, we introduce two variants of CoGenesis based on sketch and logits respectively. Our experimental findings, based on our synthesized dataset and two additional open-source datasets, indicate that: 1) Large-scale models perform well when provided with user context but struggle in the absence of such context. 2) While specialized smaller models fine-tuned on the synthetic dataset show promise, they still lag behind their larger counterparts. 3) Our CoGenesis framework, utilizing mixed-scale models, showcases competitive performance, providing a feasible solution to privacy issues.

## Dataset & Code

comming soon ...


## Citation
```latex
@inproceedings{
zhang2024cogenesis,
title={CoGenesis: A Framework Collaborating Large and Small Language Models for Secure Context-Aware Instruction Following},
author={Kaiyan Zhang, Jianyu Wang, Ermo Hua, Biqing Qi, Ning Ding, Bowen Zhou},
booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
year={2024},
url={https://arxiv.org/abs/2403.03129}
}
```
