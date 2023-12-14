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
We release our results of TimeVQVAE-AD on all 250 datasets from the UCR Time Series Anomaly archive. The results include arrays of anomaly scores and visualizations of the scores in line with the inspected time series and the labels. 
The anomaly scores have values  for each timestep of an inspected time series, therefore these scores can be utilized, for instance, to compute various metrics of any interest or to directly utilize them to compare the scores of your method's resulting anomaly scores.

In the visualizations, the color distribution and intensity of clipped $a_s^*$ is heavily dependent on the anomaly score threshold, therefore there can be some color discrepancies from the paper where the thresholds are adjusted to better showcase the examples and help the understanding.



