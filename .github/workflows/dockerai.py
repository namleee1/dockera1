#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Đọc dữ liệu
df = pd.read_excel("/kaggle/input/output1/Data system POC_11Nov2024(mod).xlsx", sheet_name="Data- refinitiv")
df_actual = pd.read_excel("/kaggle/input/output1/Data system POC_11Nov2024(mod).xlsx", sheet_name="Sheet2")

# Xử lý dữ liệu
df["Date"] = pd.to_datetime(df["Date"])
df_actual["Date"] = pd.to_datetime(df_actual["Date"])
df = df.sort_values("Date")
df.fillna(method='ffill',inplace=True)
#df = df.drop(['OMOrate', 'SBVcentralrate'], axis=1)
df_actual["VND"] = df_actual["VND"].astype(str).str.replace(" ", "").astype(float)

# Chia dữ liệu train-test
train = df[df["Date"].dt.year < 2024].drop(columns=["Date"])
test = df[df["Date"].dt.year == 2024].drop(columns=["Date"])

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)
num_features = train.shape[1]
def xgb_model(train_scaled, test_scaled, scaler, num_features):
        X_train = train_scaled[:, 1:]  
        y_train = train_scaled[:, 2]  # Cột VND
        X_test = test_scaled[:, 1:]

        model = XGBRegressor(n_estimators=200, learning_rate=0.06, max_depth= 8)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        temp = np.zeros((len(pred), num_features))  
        temp[:, 2] = pred  # Cột thứ 3 (index 2) chứa VND
        return scaler.inverse_transform(temp)[:, 2]

xgb_pred = xgb_model(train_scaled, test_scaled, scaler, num_features)

print("\n📊 Kết quả dự báo XGBoost:")
print("Ngày       | Dự báo   | Xu hướng | % Thay đổi")
print("-------------------------------------------")


# Lấy ngày cuối cùng của tập test
last_test_date = df[df["Date"].dt.year == 2024].iloc[-1]["Date"]
forecast_dates = pd.date_range(start=last_test_date, periods=n_days+1, freq="D")[1:]
for i in range(1, n_days):
    date_str = forecast_dates[i].strftime("%d-%m-%Y")
    prev_value = xgb_pred[i - 1]  # Giá trị ngày trước
    curr_value = xgb_pred[i]  # Giá trị ngày hiện tại
    change_percent = ((curr_value - prev_value) / prev_value) * 100  # % thay đổi
    trend = "📈 Up" if curr_value > prev_value else "📉 Down"

    print(f"{date_str} | {curr_value:.2f} | {trend} | {change_percent:.2f}% ")

plt.figure(figsize=(12, 6))
plt.plot(forecast_dates,df_actual["VND"][:n_days].values, label = "Truth", marker='+',color= 'red')
plt.plot(forecast_dates, xgb_pred[:n_days], label="XGBoost", linestyle="dashed", color="purple")
plt.xlabel("Ngày")
plt.ylabel("Tỷ giá VND-USD")
plt.title("Dự báo tỷ giá VND-USD")
plt.legend()
plt.show()

