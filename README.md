

# Evolutionary Graph Transformer Architecture Search Framework

Source code for the paper "**[Automatic Graph Topology-Aware Transformer]**"


We propose an evolutionary graph Transformer architecture search framework (EGTAS) to automate the construction of strong graph Transformers. We build a comprehensive graph Transformer search space with the micro-level and macro-level designs. EGTAS evolves graph Transformer topologies at the macro level and graph-aware strategies at the micro level.



## Running

- 1. 'con_data': generate the surrogate data， 
- 2. 'search': evolutionary search with surrogate model, 
- 3. 'finetune': retrain for the best arches searched by EGTAS. 
- 4. 'con_surr': construct surrogate model for Ablation studies (options),

Please see details in our code annotations.
    ```
    python main.py
    ```

## Requirements
- Python 3.x
- pytorch >=1.5.0
- torch-geometric >=2.0.3
- transformers >= 4.8.2
- tensorflow >= 2.3.1
- scikit-learn >= 0.23.2
- ogb >= 1.3.2
- datasets >=1.8.0
- geatpy >= 2.7.0

    ```
    conda create --name new_env --file packages.txt
    ```

## Results
Please refer to our paper.

## Reference
Please cite the paper whenever our proposed EGTAS is used to produce published results or incorporated into other software:
```
@article{chao,
  title={Automatic Graph Topology-Aware Transformer},
  author={Wang, Chao and Zhao, Jiaxuan and Li, Lingling and Jiao, Licheng and Liu, Fang and Yang, Shuyuan},
  journal={IEEE TNNLS(Under Review)},
  year={2023}
}
```


