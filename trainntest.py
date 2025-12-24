import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# --- 1. ตั้งค่าและโหลดข้อมูล ---
target_filename = 'Ready.xlsx'
output_filename = 'Forecast_Daily_Class_Logic.xlsx'

# 📌 กำหนด Service Level ตาม Class (Key Logic)
# Z-Score: 2.33=99% (Safety สูง), 1.645=95% (Safety กลาง), 1.28=90% (Safety ต่ำ)
SERVICE_LEVEL_MAP = {
    'A': 2.33,  
    'B': 1.645, 
    'C': 1.28
}
DEFAULT_Z = 1.645 # กรณีหา Class ไม่เจอ หรือไม่ใช่ A,B,C ให้ใช้ค่ากลางๆ

# Auto-detect file
if not os.path.exists(target_filename):
    files = [f for f in os.listdir() if f.endswith('.xlsx') or f.endswith('.csv')]
    if files: 
        target_filename = files[0]
    else: 
        print("❌ ไม่พบไฟล์ข้อมูล (.xlsx หรือ .csv)")
        exit()

print(f"✅ อ่านข้อมูลจาก: {target_filename}")
if target_filename.endswith('.csv'):
    df = pd.read_csv(target_filename)
else:
    df = pd.read_excel(target_filename)

df['requisition date'] = pd.to_datetime(df['requisition date'])

# --- 2. 🔥 แปลงข้อมูลเป็นรายวัน (Daily Aggregation) ---
print("🔄 กำลังแปลงข้อมูลเป็นรายวัน (Daily Resampling)...")

all_daily_data = []
materials = df['Material'].unique()

for mat in materials:
    sub_df = df[df['Material'] == mat].set_index('requisition date')
    
    # Resample เป็นรายวัน
    daily = sub_df.resample('D').agg({
        'x1_Quantity': 'sum',
        'x3_Class': 'last' 
    }).fillna(0)
    
    daily['Material'] = mat
    daily['Date'] = daily.index
    # Fill NA Class
    daily['x3_Class'] = daily['x3_Class'].fillna(method='ffill').fillna(method='bfill')
    
    all_daily_data.append(daily)

df_daily = pd.concat(all_daily_data).reset_index(drop=True)

# --- 3. Feature Engineering ---
print("⚙️ สร้าง Features...")
df_daily = df_daily.sort_values(by=['Material', 'Date'])

# 3.1 เวลา
df_daily['day_of_week'] = df_daily['Date'].dt.dayofweek 
df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype(int) 
df_daily['month'] = df_daily['Date'].dt.month
df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['month']/12)

# 3.2 Lag Features
lags = [1, 7, 14, 28] 
for lag in lags:
    df_daily[f'Qty_Lag{lag}'] = df_daily.groupby('Material')['x1_Quantity'].shift(lag)

# 3.3 Rolling Stats
df_daily['Roll_Mean_7D'] = df_daily.groupby('Material')['x1_Quantity'].transform(lambda x: x.shift(1).rolling(7).mean())
df_daily['Roll_Mean_30D'] = df_daily.groupby('Material')['x1_Quantity'].transform(lambda x: x.shift(1).rolling(30).mean())
df_daily['Roll_Std_30D'] = df_daily.groupby('Material')['x1_Quantity'].transform(lambda x: x.shift(1).rolling(30).std())

df_daily = df_daily.dropna()

# เก็บ Class แบบ String ไว้ใช้ตอนคำนวณ Safety Stock
df_daily['Class_Str'] = df_daily['x3_Class'].astype(str) 
df_daily['x3_Class'] = df_daily['x3_Class'].astype('category')

# --- 4. Train/Test Split ---
split_date = pd.Timestamp('2024-05-01')

features = ['day_of_week', 'is_weekend', 'month_sin', 'x3_Class',
            'Qty_Lag1', 'Qty_Lag7', 'Qty_Lag14', 'Qty_Lag28',
            'Roll_Mean_7D', 'Roll_Mean_30D', 'Roll_Std_30D']
target = 'x1_Quantity'

print(f"\n🚀 เริ่มเทรนโมเดล (ปรับ Safety Stock ตาม Class A, B, C)...")
print("-" * 60)

prediction_results = []
metrics = []

for mat in materials:
    data_mat = df_daily[df_daily['Material'] == mat]
    
    train = data_mat[data_mat['Date'] < split_date]
    test = data_mat[data_mat['Date'] >= split_date]
    
    if len(train) < 30 or len(test) == 0:
        continue

    # ดึง Class ของ Material นี้ออกมา (เอาค่าล่าสุด)
    current_class = str(data_mat['Class_Str'].iloc[-1]).strip().upper() # ทำเป็นตัวพิมพ์ใหญ่ A,B,C

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Model
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=['x3_Class'])
    
    params = {
        'objective': 'tweedie',
        'metric': 'rmse',
        'tweedie_variance_power': 1.1, 
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(params, dtrain, num_boost_round=500)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    y_pred_mean = np.round(y_pred).astype(int)
    
    # --- Evaluation ---
    rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    mae_overall = mean_absolute_error(y_test, y_pred_mean)
    
    nonzero_mask = y_test > 0
    if np.sum(nonzero_mask) > 0:
        rmse_nonzero = np.sqrt(mean_squared_error(y_test[nonzero_mask], y_pred_mean[nonzero_mask]))
        mae_nonzero = mean_absolute_error(y_test[nonzero_mask], y_pred_mean[nonzero_mask])
        rmse_nz_str = f"{rmse_nonzero:.2f}"
    else:
        rmse_nonzero = np.nan
        mae_nonzero = np.nan
        rmse_nz_str = "N/A"

    # --- 🔥 Safety Stock Logic by Class ---
    residuals = y_test - y_pred
    std_resid = np.std(residuals)
    
    # เลือกค่า Z ตาม Class (ถ้าไม่มีใน Dict ให้ใช้ค่า Default)
    selected_z = SERVICE_LEVEL_MAP.get(current_class, DEFAULT_Z)
    
    safety_stock_val = selected_z * std_resid
    recommended_qty = y_pred + safety_stock_val
    
    safety_stock_int = int(np.ceil(safety_stock_val))
    recommended_qty_int = np.ceil(recommended_qty).astype(int)

    print(f"📦 {str(mat)[:10]}.. [Class {current_class}] : RMSE(NZ)={rmse_nz_str} | Safety={safety_stock_int} (Z={selected_z})")

    # Store Results
    res_df = test[['Date', 'Material', 'x1_Quantity']].copy()
    res_df = res_df.rename(columns={'x1_Quantity': 'Actual_Qty'})
    res_df['Predicted_Mean'] = y_pred_mean
    res_df['Class'] = current_class # ใส่ Class ให้ดูด้วย
    res_df['Safety_Stock'] = safety_stock_int
    res_df['Recommended_Plan'] = recommended_qty_int
    
    prediction_results.append(res_df)
    
    metrics.append({
        'Material': mat, 
        'Class': current_class,
        'Service_Level_Z': selected_z,
        'Overall_RMSE': rmse_overall,
        'Overall_MAE': mae_overall,
        'NonZero_RMSE': rmse_nonzero,
        'NonZero_MAE': mae_nonzero,
        'Safety_Stock_Avg': safety_stock_int
    })

# --- 5. Save ---
if prediction_results:
    final_df = pd.concat(prediction_results)
    final_metrics = pd.DataFrame(metrics).sort_values(['Class', 'Overall_RMSE'], ascending=True)
    
    print("-" * 60)
    print(f"💾 บันทึกไฟล์: {output_filename}")
    
    with pd.ExcelWriter(output_filename) as writer:
        final_df.to_excel(writer, sheet_name='Daily_Plan_By_Class', index=False)
        final_metrics.to_excel(writer, sheet_name='Metrics_By_Class', index=False)
else:
    print("❌ ไม่ข้อมูลเพียงพอ")