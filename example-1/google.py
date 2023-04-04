import yfinance as yf
import matplotlib.pyplot as plt 
import ruptures as rpt
import numpy as np

# Set the start and end date
start_date = '2020-01-01'
end_date = '2022-01-01'

# Set the ticker
ticker = 'GOOGL'

# Get the data
data = yf.download(ticker, start_date, end_date)

# Print the last 5 rows
print(data.tail())


signal = np.diff(data["Open"].to_numpy())
algo = rpt.Pelt(model="rbf").fit(signal)
result = np.array(algo.predict(pen=5)) + 1


# display
rpt.display(data["Open"].to_numpy(), list(result))
plt.show()