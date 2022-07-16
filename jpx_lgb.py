import numpy as np
import pandas as pd
import jpx_tokyo_market_prediction
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
from decimal import ROUND_HALF_UP, Decimal
from scipy.special import comb
from itertools import combinations

# Loading Stock Prices
path = "../input/jpx-tokyo-stock-exchange-prediction/"
df_prices = pd.read_csv(f"{path}train_files/stock_prices.csv")
df_prices = df_prices[~df_prices["Target"].isnull()]
prices = pd.read_csv(f"{path}supplemental_files/stock_prices.csv")
df_prices = pd.concat([df_prices, prices])
#testing

def fe(df):
    df = df.sort_values(by=['Date','SecuritiesCode']).set_index('RowId')
    df['x1'] = (df['High']-df['Low'])/(df['Close']-df['Open']+0.001)
    df['x2'] = (df['High']-df['Open'])/(df['High']-df['Close']+0.001)
    df['x3'] = (df['Low']-df['Open'])/(df['Low']-df['Close']+0.001)
    df['x4'] = (df['High']-df['Low'])/(df['High']+df['Low']+0.001)
    df['x5'] = (df['Close']-df['Open'])/(df['High']+df['Low']+0.001)
    df['x6'] = (df['High']-df['Open'])/(df['High']+df['Low']+0.001)
    tlist = [1,2,3]
    df_pivot = df.pivot('Date','SecuritiesCode','Close')
    tmp = df_pivot.rolling(tlist[0]).mean().unstack().reset_index()
    tmp.columns = ['SecuritiesCode','Date',f'close_mean_{tlist[0]}']
    for tt in tlist[1:]:
        tmp2 = df_pivot.rolling(tt).mean().unstack().reset_index()
        tmp2.columns = ['SecuritiesCode','Date',f'close_mean_{tt}']
        tmp[f'close_mean_{tt}'] = tmp2[f'close_mean_{tt}']

    df = pd.merge(df,tmp,how='left',on=['Date','SecuritiesCode'])
    for t in [1,2,3]:
        df[f'close_mean_ratio_{t}'] = np.log(df.Close/df[f'close_mean_{t}'])
        df[f'close_ratio_{t}'] = np.log(df.Close/df.groupby('SecuritiesCode')['Close'].shift(t))
        del df[f'close_mean_{t}']
        
        
    df['vwap'] = (df['High']+df['Low']+df['Close']+df['Open'])/4
    df['amount'] = df['Volume']*df['vwap']

    return df

df_prices = fe(df_prices)

df_prices = df_prices[(df_prices.Date<'2022-12-06')&(df_prices.Date>'2021-01-01')]  
df_prices.info(show_counts=True)

def fill_nans(prices):
    prices.set_index(["SecuritiesCode", "Date"], inplace=True)
    prices.ExpectedDividend.fillna(0,inplace=True)
    prices.ffill(inplace=True)
    prices.fillna(0,inplace=True)
    prices.reset_index(inplace=True)
    return prices

df_prices = fill_nans(df_prices)
pd.options.display.float_format = '{:,.6g}'.format

# Utilities 

def calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    weights_mean = weights.mean()
    df = df.sort_values(by='Rank')
    purchase = (df['Target'][:portfolio_size]  * weights).sum() / weights_mean
    short    = (df['Target'][-portfolio_size:] * weights[::-1]).sum() / weights_mean
    return purchase - short

def calc_spread_return_sharpe(df, portfolio_size=200, toprank_weight_ratio=2):
    grp = df.groupby('Date')
    min_size = grp["Target"].count().min()
    if min_size<2*portfolio_size:
        portfolio_size=min_size//2
        if portfolio_size<1:
            return 0, None
    buf = grp.apply(calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio, buf

def add_rank(df, col_name="pred"):
    df["Rank"] = df.groupby("Date")[col_name].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

## By Yuike - https://www.kaggle.com/code/ikeppyo/examples-of-higher-scores-than-perfect-predictions

# This function adjusts the predictions so that the daily spread return approaches a certain value.
        
def adjuster(df):
    def calc_pred(df, x, y, z):
        return df['Target'].where(df['Target'].abs() < x, df['Target'] * y + np.sign(df['Target']) * z)

    def objective(trial, df):
        x = trial.suggest_uniform('x', 0, 0.2)
        y = trial.suggest_uniform('y', 0, 0.05)
        z = trial.suggest_uniform('z', 0, 1e-3)
        df["Rank"] = calc_pred(df, x, y, z).rank(ascending=False, method="first") - 1 
        return calc_spread_return_per_day(df, 200, 2)

    def predictor_per_day(df):
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SD))#5187
        study.optimize(lambda trial: abs(objective(trial, df) - 3), 3)
        return calc_pred(df, *study.best_params.values())

    return df.groupby("Date").apply(predictor_per_day).reset_index(level=0, drop=True)

def _predictor_base(feature_df):
    return model.predict(feature_df[feats])

def _predictor_with_adjuster(feature_df):
    df_pred = feature_df.copy()
    df_pred["Target"] = model.predict(feature_df[feats])
    return adjuster(df_pred).values.T

from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold

kfold = KFold(n_splits = 5, random_state = 2021, shuffle = True)
enumsplit = []
for fold, (trn_ind, val_ind) in enumerate(kfold.split(df_prices)):
    enumsplit.append([trn_ind, val_ind])

np.random.seed(0)
import lightgbm as lgb

feats = ['Open', 'High', 'Low', 'Close','AdjustmentFactor','ExpectedDividend',\
         'SupervisionFlag', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'close_mean_ratio_1',
       'close_ratio_1', 'close_mean_ratio_2', 'close_ratio_2',
       'close_mean_ratio_3', 'close_ratio_3', 'vwap', 'amount']

max_score = 0
params = {
    'learning_rate':0.05,
    "objective": "regression",
    "metric": "rmse",
    'boosting_type': "gbdt",
    'verbosity': -1,
    'n_jobs': -1, 
    'seed': 2030,
    'lambda_l1': 0.1002, 
    'lambda_l2': 0.1002, 
    'num_leaves': 64, 
    'feature_fraction': 0.95, 
    'bagging_fraction': 0.95, 
    'bagging_freq': 7, 
    'max_depth': -1
}


oof_predictions = np.zeros(df_prices.shape[0])
predictions = np.zeros(prices.shape[0])
modellist = []
for fold, (trn_ind, val_ind) in enumerate(enumsplit):
    print(f'Training fold {fold + 1}')
    x_train, x_val = df_prices[feats].iloc[trn_ind], df_prices[feats].iloc[val_ind]
    y_train, y_val = df_prices["Target"].iloc[trn_ind], df_prices["Target"].iloc[val_ind]
    train_dataset = lgb.Dataset(x_train, y_train)
    valid_dataset = lgb.Dataset(x_val, y_val)
    model = lgb.train(
            params,
            train_set = train_dataset, 
            valid_sets = [train_dataset, valid_dataset],
            num_boost_round = 1000,
            verbose_eval=100,
            early_stopping_rounds=50,
        )
    oof_predictions[val_ind] = model.predict(x_val)
    modellist.append(model)
    

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()
data = df_prices.copy()

for prices, options, financials, trades, secondary_prices, sample_prediction in iter_test:
    prices = fill_nans(prices)
    data = data.append(prices).drop_duplicates(["SecuritiesCode", "Date"], keep="last").sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    data = fe(data)
    prices = pd.merge(prices[['SecuritiesCode','Date']],data,how='left',on=['SecuritiesCode','Date'])
    prices["pred"] = 0
    for model in modellist:
        prices["pred"]+=model.predict(prices[feats])/len(modellist)
    prices = add_rank(prices)
    rank = prices.set_index('SecuritiesCode')['Rank'].to_dict()
    sample_prediction['Rank'] = sample_prediction['SecuritiesCode'].map(rank)
    env.predict(sample_prediction)
