import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
df = pd.read_csv("C:/Users/Chinmay/OneDrive/Documents/DataCoSupplyChainDataset.csv")


df.columns = df.columns.str.upper().str.replace(' ', '_')


df = df[
    [
        'ORDER_DATE_(DATEORDERS)',
        'CATEGORY_NAME', 'CATEGORY_ID',
        'ORDER_ITEM_QUANTITY',
        'ORDER_REGION',
        'ORDER_STATUS',
        'PRODUCT_NAME', 'PRODUCT_CARD_ID',
        'DAYS_FOR_SHIPPING_(REAL)',
        'DAYS_FOR_SHIPMENT_(SCHEDULED)'
    ]
]

df['ORDER_DATE_(DATEORDERS)'] = pd.to_datetime(df['ORDER_DATE_(DATEORDERS)'])

df['ORDER_DATE'] = df['ORDER_DATE_(DATEORDERS)'].dt.date
df['ORDER_YEAR'] = df['ORDER_DATE_(DATEORDERS)'].dt.year
df['ORDER_MONTH'] = df['ORDER_DATE_(DATEORDERS)'].dt.month
df['ORDER_WEEKDAY'] = df['ORDER_DATE_(DATEORDERS)'].dt.weekday

df.drop(columns='ORDER_DATE_(DATEORDERS)', inplace=True)

sns.histplot(df['ORDER_ITEM_QUANTITY'], kde=True)
plt.title('Distribution of Order Item Quantity')
plt.show()

daily_orders = df.groupby('ORDER_DATE')['ORDER_ITEM_QUANTITY'].sum().reset_index()

plt.plot(daily_orders['ORDER_DATE'], daily_orders['ORDER_ITEM_QUANTITY'])
plt.title('Total Orders Over Time')
plt.xticks(rotation=45)
plt.show()


sns.countplot(data=df, y='ORDER_REGION')
plt.title('Orders by Region')
plt.show()


sns.countplot(data=df, y='CATEGORY_NAME')
plt.title('Orders by Category')
plt.show()


sns.countplot(data=df, y='ORDER_STATUS')
plt.title('Order Status Distribution')
plt.show()


shipping_df = df.melt(
    id_vars='ORDER_REGION',
    value_vars=[
        'DAYS_FOR_SHIPPING_(REAL)',
        'DAYS_FOR_SHIPMENT_(SCHEDULED)'
    ],
    var_name='TYPE',
    value_name='DAYS'
)

sns.barplot(data=shipping_df, x='ORDER_REGION', y='DAYS', hue='TYPE')
plt.xticks(rotation=45)
plt.title('Shipping Days Comparison')
plt.show()


Q1 = daily_orders['ORDER_ITEM_QUANTITY'].quantile(0.25)
Q3 = daily_orders['ORDER_ITEM_QUANTITY'].quantile(0.75)
IQR = Q3 - Q1

cleaned = daily_orders[
    (daily_orders['ORDER_ITEM_QUANTITY'] >= Q1 - 1.5 * IQR) &
    (daily_orders['ORDER_ITEM_QUANTITY'] <= Q3 + 1.5 * IQR)
].copy()

cleaned['ORDER_DATE'] = pd.to_datetime(cleaned['ORDER_DATE'])

cleaned['YEAR_WEEK'] = cleaned['ORDER_DATE'].dt.to_period('W')

weekly_orders = cleaned.groupby('YEAR_WEEK')['ORDER_ITEM_QUANTITY'].sum().reset_index()
weekly_orders['ds'] = weekly_orders['YEAR_WEEK'].dt.to_timestamp()
weekly_orders.rename(columns={'ORDER_ITEM_QUANTITY': 'y'}, inplace=True)

weekly_orders = weekly_orders[['ds', 'y']]

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Train-test split
split = int(len(weekly_orders) * 0.8)
train = weekly_orders[:split]
test = weekly_orders[split:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=len(test), freq='W')
forecast = model.predict(future)

y_pred_train = forecast['yhat'][:split]
y_pred_test = forecast['yhat'][split:]

print("Train MAE:", mean_absolute_error(train['y'], y_pred_train))
print("Test MAE:", mean_absolute_error(test['y'], y_pred_test))

model.plot(forecast)
plt.title("Forecast")
plt.show()

plt.plot(train['ds'], train['y'], label='Train')
plt.plot(test['ds'], test['y'], label='Test')
plt.plot(train['ds'], y_pred_train, '--', label='Train Pred')
plt.plot(test['ds'], y_pred_test, '--', label='Test Pred')
plt.legend()
plt.show()

forecast['std_dev'] = forecast['yhat'].rolling(7).std()
forecast['avg_demand'] = forecast['yhat'].rolling(7).mean()

forecast['safety_stock'] = 1.65 * forecast['std_dev']
forecast['reorder_point'] = forecast['avg_demand'] + forecast['safety_stock']

plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
plt.plot(forecast['ds'], forecast['reorder_point'], '--', label='Reorder Point')
plt.legend()
plt.show()