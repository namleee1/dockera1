# Sử dụng một image cơ bản
FROM node:14

# Tạo thư mục làm việc trong container
WORKDIR /app

# Sao chép file package.json và package-lock.json vào thư mục làm việc
COPY package*.json ./

# Cài đặt các dependencies
RUN npm install

# Sao chép toàn bộ mã nguồn vào thư mục làm việc
# Nhập số tuần dự báo

#n_weeks = int(input("Nhập số tuần muốn dự báo: "))
#n_days = n_weeks * 7  # Chuyển thành số ngày

# Đọc dữ liệu
df = pd.read_excel("/kaggle/input/output1/Data system POC_11Nov2024(mod).xlsx", sheet_name="Data- refinitiv")
df_actual = pd.read_excel("/kaggle/input/output1/Data system POC_11Nov2024(mod).xlsx", sheet_name="Sheet2")

# Xử lý dữ liệu
df["Date"] = pd.to_datetime(df["Date"])
df_actual["Date"] = pd.to_datetime(df_actual["Date"])
df = df.sort_values("Date")
df.fillna(method='ffill')
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
                    
        model = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth= 10)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        temp = np.zeros((len(pred), num_features))  
        temp[:, 2] = pred  # Cột thứ 3 (index 2) chứa VND
        return scaler.inverse_transform(temp)[:, 2]
                
xgb_pred = xgb_model(train_scaled, test_scaled, scaler, num_features)
    
print("\n📊 Kết quả dự báo XGBoost:")
print("Ngày       | Dự báo   | Xu hướng | % Thay đổi")
print("-------------------------------------------")
    
for i in range(1, n_days):
    forecast_dates = pd.date_range(start=df["Date"].iloc[-1] , periods=n_days, freq='D')
    date_str = forecast_dates[i].strftime("%d-%m-%Y")
    prev_value = xgb_pred[i - 1]  # Giá trị ngày trước
    curr_value = xgb_pred[i]  # Giá trị ngày hiện tại
    change_percent = ((curr_value - prev_value) / prev_value) * 100  # % thay đổi
    trend = "📈 Up" if curr_value > prev_value else "📉 Down"
    
    print(f"{date_str} | {curr_value:.2f} | {trend} | {change_percent:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(forecast_dates,df_actual["VND"][:n_days].values, label = "Truth", marker='+',color= 'red')
plt.plot(forecast_dates, xgb_pred[:n_days], label="XGBoost", linestyle="dashed", color="purple")
plt.xlabel("Ngày")
plt.ylabel("Tỷ giá VND-USD")
plt.title("Dự báo tỷ giá VND-USD")
plt.legend()
plt.show()

# Mở cổng mà ứng dụng sẽ chạy
EXPOSE 3000

# Chạy ứng dụng khi container khởi động
CMD ["python", "streamlitAI.py"]
