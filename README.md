# DicTED: Dictionary Temporal Graph Network via Pre-training Embedding Distillation

> The paper based on this project has been published in ICIC 2024 (CCF-C).

> In the field of temporal graph learning, the dictionary temporal graph network is an emerging and significant technology. Compared to existing methods, dictionary networks offer greater flexibility in storage, updating, and computation during training, leading to superior performance. However, they still face some challenges: (1) These dictionary networks heavily rely on reliable raw features, and the absence of such features can result in cold start issues; (2) During the continuous updating of embeddings, catastrophic forgetting may occur. To address these issues, we propose a Dictionary Temporal Graph Network via Pre-training Embedding Distillation (DicTED). DicTED enhances the reliability of node embeddings and balances new and old knowledge by introducing a pre-trained teacher model to generate prior embeddings, combined with contrastive loss.

> Visit the [Project Homepage(Chinese)](https://samer-hue.github.io/DicTED/README.html) and [Full Paper](https://link.springer.com/chapter/10.1007/978-981-97-5678-0_29).

## Technical Background and Challenges

Graph structures, as a powerful mathematical tool, are widely applied in domains such as social networks, biological networks, recommendation systems, and knowledge graphs. Temporal graph learning extends traditional graph learning by incorporating a temporal dimension to better analyze the dynamic evolution of graphs. However, due to the limitations of data structures, obtaining rich and diverse information from different perspectives during training presents a significant challenge. This often results in increased complexity when existing methods attempt to acquire multi-source information. Furthermore, frequent updates and training can lead to memory issues.

To address these challenges, dictionary temporal graph networks have emerged. This approach aggregates neighborhood information and stores it in a dictionary format, allowing for flexible storage, updating, and retrieval when needed. However, this method still faces the following issues: (1) Dictionary networks are overly dependent on reliable raw features, and the lack of such features can lead to cold start problems. During the initialization phase, if a reliable feature set is not available, the model struggles to achieve good optimization in the early stages of training, negatively impacting subsequent training performance. (2) During the continuous updating of embeddings, catastrophic forgetting may occur. As new knowledge continuously overwrites old knowledge, the model gradually loses grasp of early information during later stages of training, affecting the comprehensive acquisition of information.

## Solution
To tackle the aforementioned issues, we pose a critical question: If introducing techniques to address the problems in dictionary temporal graph networks might introduce more complex modules, which contradicts the original intent of dictionary networks, can these issues be resolved by incorporating external prior knowledge? Motivated by this, we propose a novel approach to enhance dictionary temporal graph networks through pre-training embedding distillation, called **DicTED**. Although the knowledge distillation paradigm has been widely applied in existing work, it has yet to be extended to the field of dictionary temporal graph networks, providing us with an opportunity to fill this gap. This approach effectively addresses cold start and catastrophic forgetting issues.

We introduce multiple pre-trained teacher models to generate embeddings and integrate them as prior features for DicTED. Specifically, to address the aforementioned issues:
- **At the Input Stage:** We combine prior features with raw features to enhance the model's initialization, enabling the model to gain better information and perspectives during training.
- **At the Optimization Stage:** We align the training node embeddings with prior features through embedding loss and prediction score loss, thereby preserving the original information to a certain extent.

![DicTED Model Framework](image/framework.png)
![DicTED Model Pseudocode](image/pseudocode.png)

## Experimental Results

We validated the effectiveness of DicTED through experiments on multiple real-world datasets. The experiments focused on link prediction as the target task, using AUC (Area Under the Curve) and AP (Average Precision) as evaluation metrics. The information on the datasets and the experimental results are presented in the following tables:

![DicTED Dataset Information](image/dataset_detail.png)
![DicTED Main Experimental Results](image/performance.png)

We also conducted ablation studies and sensitivity analyses:

![DicTED Ablation Study Results](image/ablation.png)
![DicTED Sensitivity Analysis Results](image/sensitivity.png)

## Code Explanation

### Runtime Environment

We executed this project on the same device, equipped with a 22 vCPU AMD EPYC 7T83 64-core processor and an RTX 4090 GPU with 24GB of memory. The other dependencies are as follows:

```
PyTorch >= 1.4
python >= 3.7
pandas==1.4.3
tqdm==4.41.1
numpy==1.23.1
scikit_learn==1.1.2
```

### Running the Project

If you wish to run the project, the simplest way is to execute the following command, which will run the program using the default parameters specified in `parser.py`:

```
python main.py
```

If you want to customize each parameter, refer to `parser.py`. For example, you can run the following command:

```
python main.py -d wikipedia --n_hop 2 --run 2 --a1 0.3 --a2 0.3 --a3 0.2
```

### References

We introduced some pre-trained teacher models for embedding distillation. Here are their open-source codes:

```
https://github.com/tkipf/gae
https://github.com/bdy9527/SDCN
https://github.com/twitter-research/tgn
https://github.com/WenZhihao666/TREND
https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network
```

## Cite

If you use this code in your own work, please cite our paper:

```
@inproceedings{liu2024dictionary,
  title={Dictionary Temporal Graph Network via Pre-training Embedding Distillation},
  author={Liu, Yipeng and Zheng, Fang},
  booktitle={International Conference on Intelligent Computing},
  pages={336--347},
  year={2024},
  organization={Springer}
}
```
