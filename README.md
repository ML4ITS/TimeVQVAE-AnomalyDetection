# TimeVQVAE-AD

This is an official GitHub repository for the PyTorch implementation of TimeVQVAE from our paper ["Explainable Time Series Anomaly Detection using Masked Latent Generative Modeling", 2023](https://arxiv.org/abs/2311.12550).

TimeVQVAE-AD is a novel time series anomaly detection method, notable for its high accuracy and superior explainability. It builds on the TimeVQVAE method, using masked generative modeling for time-frequency domain analysis. This approach preserves dimensional semantics in the latent space, enabling precise anomaly score computation across different frequency bands. TimeVQVAE-AD also generates counterfactuals, enhancing explainability by presenting likely normal states for detected anomalies. Its effectiveness is demonstrated on the UCR Time Series Anomaly archive, outperforming existing methods in both detection accuracy and explainability.

<p align="center">
<img src=".fig/overview_inference_process_timevqvae-ad.png" alt="" width=100% height=100%>
</p>

<p align="center">
<img src=".fig/example_two_perspectives.png" alt="" width=100% height=100%>
</p>

<p align="center">
<img src=".fig/result_table.png" alt="" width=100% height=100%>
</p>

## Availability of Our Results
We release our results of TimeVQVAE-AD on all 250 datasets from the UCR Time Series Anomaly archive in `/released_results`. The results include 
1) arrays of anomaly scores in `/released_results/resulting_anomaly_score_data`, 
2) visualizations of the scores in line with the inspected time series and the labels in `/released_results/visualizations`. 

The anomaly score file, saved as a .pkl file, includes scores for each step in a time series. You can use these scores for different purposes depending on your needs (_e.g.,_ computing various metrics, comparison with your anomaly scores, etc). Our easy-to-follow Jupyter notebook tutorial, found at `/released_results/how_to_plot_anomaly_scores.ipynb`, explains the details of this data and shows you how to use it. It guides you through the process of creating the visualizations that display the anomaly scores along with the time series data.

`/released_results/visualizations` contains the detailed visualizations containing $a_s^*$, $\bar{a}_s^*$, $\bar{\bar{a}}_s^*$, and $a_{final}$. 
It should be noted that the color distribution and intensity of clipped $a_s^*$ are heavily dependent on the anomaly score threshold, therefore there can be some color discrepancies from the paper where the thresholds are adjusted to better showcase the examples for easier understanding.

[comment]: <> (We hope that our released results can benefit the community by allowing more confidence and transparency in our method.)


## Availability of TimeVQVAE-AD Implementation
The paper is currently under review, therefore the implementation code of TimeVQVAE-AD will be released as soon as the paper gets accepted. 

If you would like to implement TimeVQVAE-AD in the meantime, you may utilize [the TimeVQVAE repository](https://github.com/ML4ITS/TimeVQVAE) as a base.


