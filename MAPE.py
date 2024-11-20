# MAPE CODE
from sklearn.metrics import mean_absolute_percentage_error

# Predict the test data
y_pred = grid_search.best_estimator_.predict(X_test)
# Compute the MAPE
mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
# Get current stock price (latest 'Close' price from the data)
current_price = indicators_instock['Close'].iloc[-1]
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Current Stock Price: {current_price:.2f}")

#MAPE--> 2.93%
