# Our Results are Available Here

We release our results of TimeVQVAE-AD on all 250 datasets from the UCR Time Series Anomaly archive in `/.released_results`. The results include
1) resulting anomaly score data with relevant meta data,
2) a jupyter notebook tutorial for utilization of the resulting data.
3) visualizations of the anomaly scores in line with the inspected time series and the labels, 
4) periods of all datasets.

The anomaly score file, saved as a .pkl file in `/.released_results/resulting_anomaly_score_data`, includes scores for each step in a time series. You can use these scores for different purposes depending on your needs (_e.g.,_ computing various metrics, comparison with your anomaly scores, etc).

Our easy-to-follow Jupyter notebook tutorial, found at `/.released_results/how_to_plot_anomaly_scores_using_resulting_data.ipynb`, explains the details of this data and shows you how to use it. It guides you through the process of creating the visualizations that display the anomaly scores along with the time series data.

`/.released_results/visualizations` contains the detailed visualizations containing $a\_s^\*$, $\bar{a}\_s^\*$, $\bar{\bar{a}}\_s^\*$, and $a\_\text{final}$ for all datasets. 
It should be noted that the color distribution and intensity of clipped $a\_s^\*$ are heavily dependent on the anomaly score threshold, therefore there can be some color discrepancies from the paper where the thresholds are adjusted to better showcase the examples for easier understanding.

Moreover, we release the periods of all datasets in `/.released_results/UCR_anomaly_dataset_periods.csv`. We manually and carefully measured a period of each dataset using a plot digitizer. We manually did it instead of using the autocorrelation function because there are many datasets where the autocorrealtion cannot properly compute the period. This data is important to allow follow-up papers to use the same window size configurations. Note that we define the window size as $2 \times P$ where $P$ denotes a period.