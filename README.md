Official implementation of WWW 2024 paper "Cooperative Classification and Rationalization for Graph Generalization".

## Data download
- Spurious-Motif: this dataset can be generated via `spmotif_gen/spmotif.ipynb` in [DIR](https://github.com/Wuyxin/DIR-GNN/tree/main). 
- [MNIST-75sp](https://github.com/bknyaz/graph_attention_pool): this dataset can be downloaded [here](https://drive.google.com/drive/folders/1Prc-n9Nr8-5z-xphdRScftKKIxU4Olzh).
- Open Graph Benchmark (OGBG): this dataset can be downloaded when running c2r.sh.


## How to run C2R?

To train C2R on OGBG dataset:

```python
sh c2r.sh
```

To train C2R on Spurious-Motif dataset:

```python
# cd spmotif_codes
sh c2r.sh
```

To train C2R on MNIST-75sp dataset:

```python
# cd mnist_codes
sh c2r.sh
```

## Citation
```
@inproceedings{yue2024cooperative,
  title={Cooperative Classification and Rationalization for Graph Generalization},
  author={Yue, Linan and Liu, Qi and Liu, Ye and Gao, Weibo and Yao, Fangzhou and Li, Wenfeng},
  booktitle={Proceedings of the ACM Web Conference},
  volume={2024},
  year={2024}
}
```
