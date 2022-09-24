# JPX Tokyo Stock Exchange Prediction

Success in any financial market requires one to identify solid investments. When a stock or derivative is undervalued, it makes sense to buy. If it's overvalued, perhaps it's time to sell. While these finance decisions were historically made manually by professionals, technology has ushered in new opportunities for retail investors. Data scientists, specifically, may be interested to explore quantitative trading, where decisions are executed programmatically based on predictions from trained model.

There are plenty of existing quantitative trading efforts used to analyze financial markets and formulate investment strategies. To create and execute such a strategy requires both historical and real-time data, which is difficult to obtain especially for retail investors. This competition will provide financial data for the Japanese market, allowing retail investors to analyze the market to the fullest extent.

The competition will involve building portfolios from the stocks eligible for predictions (around 2,000 stocks). Specifically, I would predict the ranks the stocks from highest to lowest expected returns and is evaluated on the difference in returns between the top and bottom 200 stocks. The data sources contains financial data from the Japanese market, such as stock information and historical stock prices to train and test the model.

## Description

•	Proposed Long-Short Equity quantitative trading strategy through systematically selecting top and bottom expected-return performers from a 2000-stock universebased on stock information, historical stock prices data and real-time trading data 

•	Constructured features such as price movement, volatility, increased volume ratio, and VWAP to identify the potential ML models

•	Trained machine learning models including LightGBM, Catboost and Tabnet based on cross-validation RMSE and  assigned equal ensemble weighting on aggregated prediction results to reduce overfitting and improved out-of-sample prediction accuracy by 12% 

•	Rebalanced portfolio dynamically based on daily stock performance ranking predictionand continuously evaluated on new streaming out-of-sample data within future three-month window; achived Sharp Ratio around 1.3

## Getting Started

### Dependencies

* Kaggle Notebook

## Authors
Siquan Wang

sw3442@cumc.columbia.edu

## Version History

* 0.2
    * Various bug fixes and optimizations
    * Final version
* 0.1
    * Initial Release 

## License

N/A

## Acknowledgments

competition website:
* https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction
