#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import VAR
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import streamlit as st


# Tải file lên Streamlit
uploaded_file = st.file_uploader("/kaggle/input/datastreamlit/Data system POC_11Nov2024(mod).xlsx", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Đọc file Excel
    df = pd.read_excel(uploaded_file, sheet_name="Data- refinitiv")
    df1 = pd.read_excel(uploaded_file, sheet_name="Sheet2")

    plt.figure(figsize=(11,7)) # Kích cỡ biểu đồ
    plt.grid(True)


    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["FEDRATE"], marker= '+', color= 'b' )
    ax.set_title("FEDRATE CHANGES AFTER YEARS")
    ax.set_xlabel("DATE")
    ax.set_ylabel("FEDRATE")
    ax.legend()

    st.pyplot(fig)

    plt.figure(figsize=(11,7)) # Kích cỡ biểu đồ
    plt.grid(True)


    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["DXY"], marker= '+', color= 'b' )
    ax.set_title("DXY CHANGES AFTER YEARS")
    ax.set_xlabel("DATE")
    ax.set_ylabel("DXY")
    ax.legend()

    st.pyplot(fig)

    plt.figure(figsize=(11,7)) # Kích cỡ biểu đồ
    plt.grid(True)
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["VND"], marker= '+', color= 'b' )
    ax.set_title("VND-USD CHANGES AFTER YEARS")
    ax.set_xlabel("VND-USD")
    ax.set_ylabel("VND-USD")
    ax.legend()

    st.pyplot(fig)

    plt.figure(figsize=(11,7)) # Kích cỡ biểu đồ
    plt.grid(True)
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["OMOrate"], marker= '+', color= 'b' )
    ax.set_title("OMOrate CHANGES AFTER YEARS")
    ax.set_xlabel("DATE")
    ax.set_ylabel("OMOrate")
    ax.legend()

    st.pyplot(fig)

    plt.figure(figsize=(11,7)) # Kích cỡ biểu đồ
    plt.grid(True)
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["SBVcentralrate"], marker= '+', color= 'b' )
    ax.set_title("SBVcentralrate CHANGES AFTER YEARS")
    ax.set_xlabel("DATE")
    ax.set_ylabel("SBVcentralrate")
    ax.legend()

    st.pyplot(fig)

    #hiển thị tương quan giữa các cột
    correlation_matrix = df.corr()
    st.write(correlation_matrix)

    # Người dùng nhập tên chỉ số và số ngày cần xem
    name = st.text_input("Nhập tên chỉ số muốn xem dữ liệu lịch sử (FEDRATE, DXY, VND, OMOrate, SBVcentralrate):")
    n = st.number_input("Nhập số ngày muốn xem:", min_value=1, step=1)

    if name and n:
        if name == 'FEDRATE':
            st.write(df[['Date', 'FEDRATE']].iloc[:n])
        elif name == 'DXY':
            st.write(df[['Date', 'DXY']].iloc[:n])
        elif name == 'VND':
            st.write(df[['Date', 'VND']].iloc[:n])
        elif name == 'OMOrate':
            st.write(df[['Date', 'OMOrate']].iloc[:n])
        elif name == 'SBVcentralrate':
            st.write(df[['Date', 'SBVcentralrate']].iloc[:n])
        else:
            st.warning("Chỉ số không hợp lệ. Vui lòng chọn từ các chỉ số trên.")

        # Hàm train mô hình xgb

    df["Date"] = pd.to_datetime(df["Date"])
    df1["Date"] = pd.to_datetime(df1["Date"])
    df1["VND"] = df1["VND"].astype(str).str.replace(" ", "").astype(float)
    df = df.sort_values("Date")
    #df = df.drop(['OMOrate', 'SBVcentralrate'], axis=1)  # Chỉ giữ lại các cột quan trọng

    train = df[df["Date"].dt.year < 2024].drop(columns=["Date"])
    test = df[df["Date"].dt.year == 2024].drop(columns=["Date"])
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

                # Huấn luyện mô hình XGBoost (SỬA LỖI)
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

    st.write("\n📊 Kết quả dự báo XGBoost:")
    st.write("Ngày       | Dự báo   | Xu hướng | % Thay đổi")
    st.write("-------------------------------------------")

    for i in range(1, n_days):
        forecast_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')
        date_str = forecast_dates[i].strftime("%d-%m-%Y")
        prev_value = xgb_pred[i - 1]  # Giá trị ngày trước
        curr_value = xgb_pred[i]  # Giá trị ngày hiện tại
        change_percent = ((curr_value - prev_value) / prev_value) * 100  # % thay đổi
        trend = "📈 Up" if curr_value > prev_value else "📉 Down"

    st.write(f"{date_str} | {curr_value:.2f} | {trend} | {change_percent:.2f}%")

                # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))

    plt.plot(forecast_dates, df1["VND"][:n_days].values, label="Thực tế", color="blue")
    plt.plot(forecast_dates, xgb_pred[:n_days], label="XGBoost", linestyle="dashed", color="purple")


    plt.xlabel("Ngày")
    plt.ylabel("Tỷ giá VND-USD")
    plt.title("Dự báo tỷ giá VND-USD")
    plt.legend()
    plt.show()

    st.pyplot(plt)


# 🔹 Lấy mật khẩu email từ biến môi trường
EMAIL_SENDER = "namltmta@gmail.com"
EMAIL_PASSWORD = "jlxk sqlk gckc eqzz"

# 🔹 Nhập email người nhận từ Streamlit
EMAIL_RECEIVER = st.text_input('Mời bạn nhập vào email người nhận:')

# 🔹 Kiểm tra nếu người dùng đã nhập email
if EMAIL_RECEIVER:
    # 🔹 Giả lập dữ báo tỷ giá (dữ liệu mẫu)
    forecast_dates = forecast_dates
    predictions = predictions

    # 🔹 Tạo nội dung email
    email_content = "<h2>Dự báo tỷ giá VND-USD tuần tới</h2><table border='1' cellpadding='5'>"
    email_content += "<tr><th>Ngày</th><th>Dự đoán giá</th></tr>"

    for date, price in zip(forecast_dates, predictions):
        email_content += f"<tr><td>{date}</td><td>{price:.2f} USD</td></tr>"

    email_content += "</table>"

    # 🔹 Thiết lập email
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = "Báo cáo dự báo tỷ giá VND-USD tuần tới"
    msg.attach(MIMEText(email_content, 'html'))

    # 🔹 Gửi email qua SMTP (Gmail SMTP Server)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        st.success("✅ Email đã được gửi thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi khi gửi email: {str(e)}")



# In[ ]:




