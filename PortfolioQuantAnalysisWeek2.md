```python
#Portfolio Analytics
import numpy as np
import matplotlib.pyplot as plt
import equity_analysis as ea
import seaborn as sns


data = ea.get_data(["AAPL", "MSFT", "GOOG"], start="2018-01-01", end="2023-01-01")
returns = ea.compute_returns(data)
weight = np.array([1/3,1/3,1/3])
portfolio_returns = returns @ weight
print(portfolio_returns)
cum_returns = ea.compute_cumulative_returns(returns)
print(cum_returns)
cum_portfolio = (1 + portfolio_returns).cumprod()
print(cum_portfolio)

plt.figure(figsize=[12,6])
plt.plot(cum_returns, alpha=0.8)
plt.plot(cum_portfolio, color = 'black', linewidth = 2, label = 'Portfolio')
plt.title('Portfolio returns vs Individual Stock Cumulative Returns')
plt.legend(list(returns.columns) + ['Portfolio'])
plt.show()



```

    /Users/keyansaguibo/PyCharmMiscProject/.venv/quant_projects/equity_analysis.py:6: FutureWarning: YF.download() has changed argument auto_adjust default to True
      data = yf.download(tickers, start=start, end=end)["Close"]
    [*********************100%***********************]  3 of 3 completed

    Date
    2018-01-03    0.006964
    2018-01-04    0.005689
    2018-01-05    0.012785
    2018-01-08    0.000526
    2018-01-09   -0.000470
                    ...   
    2022-12-23    0.005677
    2022-12-27   -0.014075
    2022-12-28   -0.019219
    2022-12-29    0.028251
    2022-12-30   -0.001647
    Length: 1258, dtype: float64
    Ticker          AAPL      GOOG      MSFT
    Date                                    
    2018-01-03  0.999826  1.016413  1.004654
    2018-01-04  1.004470  1.020094  1.013496
    2018-01-05  1.015906  1.034958  1.026061
    2018-01-08  1.012133  1.039380  1.027108
    2018-01-09  1.012017  1.038742  1.026410
    ...              ...       ...       ...
    2022-12-23  3.219985  1.686573  2.949454
    2022-12-27  3.175297  1.651268  2.927586
    2022-12-28  3.077862  1.623662  2.897564
    2022-12-29  3.165040  1.670423  2.977623
    2022-12-30  3.172854  1.666291  2.962921
    
    [1258 rows x 3 columns]
    Date
    2018-01-03    1.006964
    2018-01-04    1.012693
    2018-01-05    1.025640
    2018-01-08    1.026180
    2018-01-09    1.025698
                    ...   
    2022-12-23    2.626429
    2022-12-27    2.589461
    2022-12-28    2.539693
    2022-12-29    2.611443
    2022-12-30    2.607141
    Length: 1258, dtype: float64


    



    
![png](PortfolioQuantAnalysisWeek2_files/PortfolioQuantAnalysisWeek2_0_3.png)
    


Analysis:

-The ea.get_data() function retrieves daily closing prices for the selected stocks between 2018 and 2023.

-ea.compute_returns() converts price data into daily percentage returns for each stock.

-weight = np.array([1/3, 1/3, 1/3]) assigns equal portfolio weights, giving each stock one-third of the total allocation.

-The matrix multiplication returns @ weight computes the portfolio’s daily return as a weighted average of the individual stock returns.

-ea.compute_cumulative_returns() and (1 + portfolio_returns).cumprod() calculate cumulative growth for the individual assets and the total portfolio, respectively.

The resulting plot visualizes performance over time:

-Each colored line represents an individual stock’s cumulative return since 2018.

-The bold black line represents the portfolio’s cumulative return, showing how a diversified mix behaves compared to holding a single asset.

-The portfolio line is typically smoother (less volatile) than the individual stock lines, demonstrating the benefit of diversification — spreading risk reduces the impact of any one asset’s fluctuations.

My takeaway:
This analysis highlights how combining multiple assets can lead to more stable growth and lower volatility, even if the overall return is slightly lower than the best-performing individual stock.
It’s a foundational concept in modern portfolio theory: diversification improves the risk-return tradeoff.


```python
cum_returns.plot(figsize=[12,6])
cum_portfolio.plot(figsize=[12,6])
```




    <Axes: xlabel='Date'>




    
![png](PortfolioQuantAnalysisWeek2_files/PortfolioQuantAnalysisWeek2_2_1.png)
    


Analysis:

-The cum_returns variable contains the cumulative growth of each stock (AAPL, MSFT, and GOOG), showing how much each has increased since 2018.

-The cum_portfolio line represents the combined performance of all three assets, assuming equal weights (⅓ each).

-By plotting them together, we can see how diversification smooths out volatility compared to holding any single stock.

Graph Interpretation

-The portfolio curve typically lies between the best- and worst-performing stocks — a visual proof of risk reduction through diversification.

-During volatile periods (e.g., market dips), the portfolio line tends to decline less sharply than the most volatile individual stocks.

-Over time, the portfolio’s growth path appears more stable and consistent, even if it doesn’t always match the peak returns of the top-performing stock.

My takeaway:
This comparison demonstrates that diversifying across multiple assets balances returns and risk, producing a smoother performance curve. While you might give up some upside from the strongest stock, you gain more stability and downside protection overall — a core principle of Modern Portfolio Theory.


```python
#volatility portfolio
cov_matrix = returns.cov()
port_variance = weight.T @ cov_matrix @ weight
port_volatility = np.sqrt(port_variance)  # daily std
print("Daily portfolio volatility:", port_volatility)

# Individual stock vols (daily)
individual_vols = returns.std()
print("Daily individual vols:", individual_vols)



```

    Daily portfolio volatility: 0.018436386162133934
    Daily individual vols: Ticker
    AAPL    0.021094
    GOOG    0.019751
    MSFT    0.019549
    dtype: float64


Explanation of the Code:

returns.cov() computes the covariance matrix, which measures how each stock’s returns move relative to the others.

If two stocks tend to rise and fall together, their covariance will be positive.

If they move in opposite directions, the covariance will be negative, helping reduce portfolio risk.

port_variance = weight.T @ cov_matrix @ weight applies the portfolio variance formula

Taking the square root gives portfolio volatility (standard deviation) — a measure of daily risk.

Finally, returns.std() calculates each stock’s individual volatility, allowing a comparison between single-stock and portfolio risk.

Interpretation:

The portfolio volatility is typically lower than the average individual stock volatility because the assets are not perfectly correlated. This demonstrates the diversification benefit: combining assets that don’t move exactly together reduces overall portfolio risk. Even though each stock may experience sharp movements individually, the portfolio’s combined behavior is smoother and more stable.

My takeaway:
This step highlights one of the core principles of Modern Portfolio Theory (MPT) — diversification allows investors to minimize total portfolio volatility without necessarily lowering expected returns.


```python
#Sharpe Ratio
mean_returns = returns.mean()
mean_portfolio = portfolio_returns.mean()
annualized_return = mean_returns * 252
annualized_vol = individual_vols * np.sqrt(252)
annualized_portfolio = mean_portfolio * 252
annualized_portfolio_vol = port_volatility * np.sqrt(252)
sharpe_portfolio = annualized_portfolio / annualized_portfolio_vol
sharpe_individual = annualized_portfolio / annualized_vol
print("Portfolio annual return:", annualized_return)
print("Portfolio annual volatility:", annualized_portfolio_vol)
print("Individual Sharpe:", sharpe_individual)
print("Portfolio Sharpe:", sharpe_portfolio)


```

    Portfolio annual return: Ticker
    AAPL    0.287461
    GOOG    0.151470
    MSFT    0.265836
    dtype: float64
    Portfolio annual volatility: 0.2926685571585536
    Individual Sharpe: Ticker
    AAPL    0.701574
    GOOG    0.749257
    MSFT    0.757001
    dtype: float64
    Portfolio Sharpe: 0.8026905400895123



Steps:

Calculate mean daily returns for each stock and the portfolio.

Annualize returns (×252) and volatility (×√252).

Divide annualized return by annualized volatility to get the Sharpe Ratio.

Interpretation:

Sharpe > 1 → good performance

Sharpe > 2 → excellent

Portfolio Sharpe > individual Sharpe → diversification improved efficiency

Example:
A Sharpe Ratio of 0.8 means the portfolio earns 0.8 units of return for every unit of risk.


```python
#Visualization

all_sharpes = sharpe_individual.copy()
all_sharpes["Portfolio"] = sharpe_portfolio

all_sharpes.plot(kind='bar', figsize=(10,6), color='skyblue')
plt.title("Sharpe Ratios (Risk-Adjusted Performance)")
plt.ylabel("Sharpe Ratio")
plt.show()
```


    
![png](PortfolioQuantAnalysisWeek2_files/PortfolioQuantAnalysisWeek2_8_0.png)
    


Explanation:

Each bar represents the Sharpe Ratio for one asset or the overall portfolio.

The portfolio bar shows how diversification impacts performance compared to holding individual assets.

A higher bar indicates better returns per unit of risk.

Interpretation Example:
If the portfolio’s bar is taller than any single stock’s, it means the portfolio achieved superior efficiency — higher reward for the same or less risk.


```python
#Correlations Between Assets
corr_matrix = returns.corr()
print(corr_matrix)
```

    Ticker      AAPL      GOOG      MSFT
    Ticker                              
    AAPL    1.000000  0.700213  0.772959
    GOOG    0.700213  1.000000  0.804103
    MSFT    0.772959  0.804103  1.000000


Explanation:

The correlation matrix shows values between -1 and 1 for each pair of assets.

+1 → move perfectly together

0 → no relationship

-1 → move in opposite directions

Interpretation:

Lower correlations between assets (closer to 0 or negative) mean better diversification.

For example, if AAPL and MSFT have a correlation of 0.85, they tend to move similarly — so combining them provides less diversification benefit.

A lower correlation (like 0.3) means the assets don’t move together as much, reducing portfolio risk.


```python
plt.figure(figsize=[12,6])
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0 )
plt.title("Correlation Matrix of Returns")
plt.show()
```


    
![png](PortfolioQuantAnalysisWeek2_files/PortfolioQuantAnalysisWeek2_12_0.png)
    


Explanation:

The heatmap visually represents the correlation matrix:

Red areas (closer to +1) → assets move together (high correlation).

Blue areas (closer to -1) → assets move oppositely (negative correlation).

White or neutral colors (around 0) → weak or no relationship.

The diagonal will always show 1.0 (each asset perfectly correlates with itself).

Interpretation:

A portfolio of highly correlated assets offers little diversification.

A mix of low or negatively correlated assets can help reduce risk while maintaining returns.

Example:
If AAPL and MSFT appear dark red, they move closely together — whereas if AAPL and GOOG are lighter, they offer more diversification potential.


```python
sns.pairplot(returns)
plt.show()
```


    
![png](PortfolioQuantAnalysisWeek2_files/PortfolioQuantAnalysisWeek2_14_0.png)
    



```python
#efficient frontier/monte carlo
cov_matrix_annual = returns.cov() * 252
n_assets = len(mean_returns)
n_portfolios = 10000
results = np.zeros((3,n_portfolios))

for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)

    portfolio_return = np.dot(weights, annualized_return)
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix_annual @ weights)
    sharpe_ratio = portfolio_return / portfolio_volatility

    results[0,i] = portfolio_volatility
    results[1,i] = portfolio_return
    results[2,i] = sharpe_ratio

plt.figure(figsize=(10,6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap="viridis", s=10)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Return")
plt.title("Efficient Frontier Simulation")
plt.show()



```


    
![png](PortfolioQuantAnalysisWeek2_files/PortfolioQuantAnalysisWeek2_15_0.png)
    


The Efficient Frontier represents the set of portfolios that offer the maximum expected return for a given level of risk.
Using Monte Carlo simulation, we can randomly generate thousands of portfolios to visualize the trade-off between risk and return.

Explanation:

Each point on the scatter plot represents a randomly generated portfolio.

The x-axis shows portfolio volatility (risk), and the y-axis shows expected return.

The color of each point indicates the Sharpe Ratio, measuring risk-adjusted performance.

The upper boundary of the scatter cloud forms the Efficient Frontier — the best possible portfolios for each risk level.

Interpretation:

Portfolios near the top-left (high return, low risk) are optimal.

Portfolios below the frontier are inefficient — they carry more risk for the same return.

This helps investors choose the best portfolio for their risk tolerance.
