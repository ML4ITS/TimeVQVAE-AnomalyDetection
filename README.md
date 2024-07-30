# TimeVQVAE-AD

This is an official GitHub repository for the PyTorch implementation of TimeVQVAE from our paper, ["Explainable time series anomaly detection using masked latent generative modeling." Pattern Recognition (2024)](https://arxiv.org/abs/2311.12550).

TimeVQVAE-AD is a novel time series anomaly detection method, notable for its high accuracy and superior explainability. It builds on TimeVQVAE [1], using masked generative modeling for time-frequency domain analysis. This approach preserves dimensional semantics in the latent space, enabling precise anomaly score computation across different frequency bands. TimeVQVAE-AD also generates counterfactuals, enhancing explainability by presenting likely normal states for detected anomalies. Its effectiveness is demonstrated on the UCR Time Series Anomaly archive, significnatly outperforming existing methods in both detection accuracy and explainability.

<p align="center">
<img src=".fig/overview_inference_process_timevqvae-ad.png" alt="" width=100% height=100%>
</p>

<p align="center">
<img src=".fig/example_two_perspectives.png" alt="" width=100% height=100%>
</p>

<p align="center">
<img src=".fig/result_table.png" alt="" width=100% height=100%>
</p>

## Prerequisite

Environmental Setup:
...

Dataset:
download the dataset, locate it ...


## Usage

### Configuration


### Training: Stage1


### Training: Stage2


### Evaluation



## Our Results are Available in `.released_results/`
For details, see `.released_results/README.md`.


## References
[1] Lee, Daesoo, Sara Malacarne, and Erlend Aune. "Vector Quantized Time Series Generation with a Bidirectional Prior Model." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.


## Citation
```
@article{lee2024explainable,
  title={Explainable time series anomaly detection using masked latent generative modeling},
  author={Lee, Daesoo and Malacarne, Sara and Aune, Erlend},
  journal={Pattern Recognition},
  pages={110826},
  year={2024},
  publisher={Elsevier}
}
```