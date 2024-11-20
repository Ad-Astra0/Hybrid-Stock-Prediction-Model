# MAIN CODE FOR MODEL!!!!
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from transformers import BertTokenizer,TFBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from bs4 import BeautifulSoup
import requests
import datetime as dt
from scipy.stats import mode
def calculate_d1_d2(S,K,T,r,sigma):
  d1=(np.log(S/K)+r(0.5*sigma**2)*T)/(sigma*np.sqrt(T))
  d2=d1-sigma*np.sqrt(T)
  return d1,d2
def black_scholes(S,K,T,r,sigma,option_type='call'):
  if sigma==0:
    return 0
  if T==0 or S==0 or K==0:
    return 0
    d1,d2=calculate_d1_d2(S,K,T,r,sigma)
    if option_type=='call':
      price=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type=='put':
      price=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return price

def get_option_data(ticker):
  stock=yf.Ticker(ticker)
  expiration_dates=stock.options
  option_chain=stock.option_chain(expiration_dates[0])
  calls=option_chain.calls
  puts=option_chain.puts
  return calls,puts,expiration_dates[0]
def get_risk_free_rate():
    return 0.05
# FinBERT sentiment analysis: CREDIT TO FINBERT and ProsusAI
def get_news_headlines(ticker):
    url=f"https://www.yahoo.com/quote/{ticker}/news"
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    headlines=[]
    for h3 in soup.find_all('h3'):
      headlines.append(h3.text)
    return headlines[:10]

def get_sentiments(headlines):
    finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    results = finbert_pipeline(headlines)
    return [1 if result['label'] == 'positive' else 0 for result in results]
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
# Define the model for BiLSTM
class BERTBiLSTMSentiments(nn.Module):
    def __init__(self,hidden_size,num_classes):
        super().__init__()
        self.bert=TFBertModel.from_pretrained("bert-base-uncased")
        self.lstm=nn.LSTM(768,hidden_size,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_size*2,num_classes)
    def forward(self,input_ids,attention_mask):
      lstm_output,_=self.lstm(self.bert(input_ids=input_ids,attention_mask=attention_mask)[0])
      return self.fc(lstm_output[:,-1,:])
def get_sentiment(text,model,tokenizer,device):
  inputs=tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=512).to(device)
  with torch.no_grad():
    outputs=model(inputs["input_ids"],inputs["attention_mask"])
    max_outputs_index=torch.max(outputs,dim=1)
    return max_outputs_index[0].item()-1
def get_news_sentiment(ticker,model,tokenizer,device):
  headlines=get_news_headlines(ticker)
  sentiments=[]
  for headline in headlines:
    sentiment=get_sentiment(headline,model,tokenizer,device)
    sentiment.append(sentiment)
  if len(sentiments)>0:
    return np.mean(sentiments)
  else:
    return 0
  device = get_device()
  #Intitialize BERT-BiLSTM model
  model=BERTBiLSTMSentiments(hidden_size=128,num_classes=3).to(device)
  tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
  #Get news headlines
  headlines=get_news_headlines(ticker)
  #FinBERT sentiment analysis
  finbert_sentiments=get_sentiments(headlines)
  #BERT-BiLSTM sentiment analysis
  bert_bilstm_sentiments=[]
  for headline in headlines:
    sentiment=get_sentiment(headline,model,tokenizer,device)
    bert_bilstm_sentiments.append(sentiment)
    # Print results
    print(f"News headlines and sentiment for {ticker}:")
  for i in range(len(headlines)):
    print(f"Headline:",{headlines[i]})

def calculate_rsi(series,period=14):
    delta=series.diff()
    gain=(delta.where(delta>0,0)).rolling(window=period,min_periods=1).mean()
    loss=(-delta.where(delta<0,0)).rolling(window=period,min_periods=1).mean()
    rs=gain/loss
    rsi=100-(100/(1+rs))
    return rsi
def calculate_macd(series,slow=26,fast=12,signal=9):
  exp1=series.ewm(span=fast,adjust=False).mean()
  exp2=series.ewm(span=slow,adjust=False).mean()
  macd=exp1-exp2
  signal_line=macd.ewm(span=signal,adjust=False).mean()
  return macd,signal_line

#Intalialize device and model
device = get_device()
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=BERTBiLSTMSentiments(hidden_size=128,num_classes=3).to(device)
# Stock data and indicators
ticker = "NVDA"
stock = yf.Ticker(ticker)
start="2024-01-01"
end="2025-01-01"
indicators_instock = stock.history(start=start,end=end )
indicators_instock['MA20']=indicators_instock['Close'].rolling(window=20).mean()
indicators_instock['MA50']=indicators_instock['Close'].rolling(window=50).mean()
indicators_instock['RSI']=calculate_rsi(indicators_instock['Close'],14)
indicators_instock['MACD'],indicators_instock['MACD_Signal']=calculate_macd(indicators_instock['Close'])
commodities=['GC=F','CL=F','SI=F']
for commodity in commodities:
  commodity_data=yf.download(commodity,start='2020-01-01',end='2023-01-01')
  indicators_instock[f'{commodity}_Close']=commodity_data['Close']
  if not commodity_data['Close'].empty:
    indicators_instock[f'{commodity}_RSI'] = calculate_rsi(commodity_data['Close'], 14)
    indicators_instock[f'{commodity}_MACD'], _ = calculate_macd(commodity_data['Close'])
news_sentiment=get_news_sentiment(ticker,model,tokenizer,device)
indicators_instock['News_Sentiment']=news_sentiment
info = stock.info
fundamentals = {
    'Market_Cap': info.get("marketCap"),
    'Enterprise_Value': info.get('enterpriseValue'),
    'Trailing_PE': info.get('trailingPE'),
    'Forward_PE': info.get('forwardPE'),
    'PEG_Ratio': info.get('pegRatio'),
    'Price_to_Sales': info.get('priceToSalesTrailing12Months'),
    'Price_to_Book': info.get('priceToBook'),
    'EV_to_Revenue': info.get('enterpriseToRevenue'),
    'EV_to_EBITDA': info.get('enterpriseToEbitda'),
    'Profit_Margin': info.get('profitMargins'),
    'ROA': info.get('returnOnAssets'),
    'ROE': info.get('returnOnEquity'),
    'Revenue': info.get('totalRevenue'),
    'Net_Income': info.get('netIncomeToCommon'),
    'Diluted_EPS': info.get('trailingEps'),
    'Total_Cash': info.get('totalCash'),
    'Debt_to_Equity': info.get('debtToEquity'),
    'Free_Cash_Flow': info.get('freeCashflow')
}
for key, value in fundamentals.items():
    indicators_instock[key] = value
# Clean up and prepare data for model
# Drop columns with all NaN values and impute remaining NaN values with the mean
indicators_instock=indicators_instock.dropna(axis=1, how='all')
indicators_instock[:]=SimpleImputer(strategy='mean').fit_transform(indicators_instock)
#Outliers and envenelope
elliptic_env=EllipticEnvelope(contamination=0.05)
outliers=elliptic_env.fit_predict(indicators_instock.select_dtypes(include=[np.number]))
indicators_instock=indicators_instock[outliers==1]
# Prepare features and target for model training
existing_columns=indicators_instock.columns

commodity_features = []
for commodity in commodities:
    if f'{commodity}_RSI' in existing_columns:
        commodity_features.append(f'{commodity}_RSI')
    if f'{commodity}_MACD' in existing_columns:
        commodity_features.append(f'{commodity}_MACD')
target = indicators_instock['Close'].shift(-1).dropna()
features = indicators_instock.drop(columns=['Close']).iloc[:-1]
# Option data
calls, puts, exp_date = get_option_data(ticker)
S = indicators_instock['Close'].iloc[-1]
T = (dt.datetime.strptime(exp_date, '%Y-%m-%d') - dt.datetime.now()).days / 365
r = get_risk_free_rate()
# Impute implied volatility
imputer = SimpleImputer(strategy='median')
dataframes=[calls,puts]
# Apply the imputer to each DataFrame
for df in dataframes:
    df[['impliedVolatility']]=imputer.fit_transform(df[['impliedVolatility']])
# Calculate Black-Scholes prices
calls['BS_price'] = calls.apply(lambda row: black_scholes(S, row['strike'], T, r, row['impliedVolatility'], 'call'), axis=1)
puts['BS_price'] = puts.apply(lambda row: black_scholes(S, row['strike'], T, r, row['impliedVolatility'], 'put'), axis=1)
print("Call Options with Black-Scholes Prices:\n", calls[['strike', 'lastPrice', 'impliedVolatility']].head())
print("Put Options with Black-Scholes Prices:\n", puts[['strike', 'lastPrice', 'impliedVolatility']].head())

# Prepare for model training
X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=42)
pipeline=Pipeline([
    ('scaler',StandardScaler()),
    ('regressor',Ridge())
])
hyperparameters={'regressor__alpha':[0.01,0.1,1.0,10.0,100.0]}
grid_search=GridSearchCV(pipeline,hyperparameters,cv=5,scoring='r2')
grid_search.fit(X_train, y_train)
# Monte Carlo Simulation using Merton Jump-Diffusion with EVT and Asymmetry
n_simulations = 1000
n_days = 252
last_price = indicators_instock['Close'].iloc[-1]
mu=grid_search.best_estimator_.predict(X_test).mean()
sigma=grid_search.best_estimator_.predict(X_test).std()
mu=np.clip(mu,-0.2,0.2)
sigma=np.clip(sigma,0.1,0.5)
def merton_jump_diffusion_EVT_asymmetry(S0,mu,sigma,T,dt,jumps_lamda,jumps_sigma,jumps_mu,n_sims):
    n_steps=int(T/dt)
    prices=np.zeros((n_sims,n_steps))
    prices[:,0]=S0
    for t in range(1,n_steps):
      brownian_motion=np.random.normal(mu*dt,sigma*np.sqrt(dt),size=n_sims)
      poisson_jumps=np.random.poisson(jumps_lamda*dt,size=n_sims)
      jump_sizes=poisson_jumps*np.random.normal(jumps_mu,jumps_sigma,size=n_sims)
      prices[:,t]=prices[:,t-1]*np.exp(brownian_motion+jump_sizes)
    return prices
T = 1.0
dt = 1 / 252
jump_lamda = 0.15
jump_sigma = 0.15
jump_mu = -0.05
simulation_results=merton_jump_diffusion_EVT_asymmetry(last_price,mu,sigma,T,dt,jump_lamda,jump_sigma,jump_mu,n_simulations)
# Plot the simulation
average_simulation = np.median(simulation_results, axis=0)
min_price = max(25, simulation_results.min())
max_price = simulation_results.max()
# Define the percentiles you want to calculate
percentiles=[5,25,75,85]

# Calculate percentiles using a list comprehension
percentile_values={p:np.percentile(simulation_results,p,axis=0) for p in percentiles}

# Prepare the plot
plt.figure(figsize=(12,8))

plt.plot(simulation_results.T,color='skyblue',alpha=0.1)
plt.plot(average_simulation,color='navy',linewidth=2,label='Median (50th Percentile)')
# Plot each calculated percentile
for p, value in percentile_values.items():
    plt.plot(value,linewidth=2,linestyle='--',label=f'{p}th Percentile')

# Additional percentiles that are not in the loop
plt.plot(percentile_values[5], color='crimson', linewidth=2, linestyle='--', label='5th Percentile')
plt.plot(percentile_values[25], color='orange', linewidth=2, linestyle='--', label='25th Percentile')
plt.plot(percentile_values[75], color='limegreen', linewidth=2, linestyle='--', label='75th Percentile')
plt.plot(percentile_values[85], color='purple', linewidth=2, linestyle='--', label='85th Percentile')

# Customize the plot
plt.title(f"Monte Carlo Simulation for {ticker} with Merton Jump-Diffusion, EVT & Asymmetry")
plt.xlabel("Days")
plt.ylabel("Price")
plt.ylim(min_price, max_price)
plt.legend()
plt.show()
