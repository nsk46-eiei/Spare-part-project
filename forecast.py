import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# --- 1. ตั้งค่าและโหลดข้อมูล ---
target_filename = 'Ready.xlsx'
output_filename = 'Forecast_2025_Jun_Dec_Fixed.xlsx' # เปลี่ยนชื่อไฟล์ output

# Service Level Map (สำหรับ Safety Stock)
SERVICE_LEVEL_MAP = {
    'A': 2.33,   # 99%
    'B': 1.645,  # 95%
    'C': 1.28    # 90%
}
DEFAULT_Z = 1.645

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

# --- 2. แปลงข้อมูลเป็นรายวัน (Daily Aggregation) ---
print("🔄 แปลงข้อมูลเป็นรายวัน...")
all_daily_data = []
materials = df['Material'].unique()

for mat in materials:
    sub_df = df[df['Material'] == mat].set_index('requisition date')
    daily = sub_df.resample('D').agg({
        'x1_Quantity': 'sum',
        'x3_Class': 'last' 
    }).fillna(0)
    
    daily['Material'] = mat
    daily['Date'] = daily.index
    daily['x3_Class'] = daily['x3_Class'].fillna(method='ffill').fillna(method='bfill')
    all_daily_data.append(daily)

df_daily = pd.concat(all_daily_data).reset_index(drop=True)
df_daily = df_daily.sort_values(by=['Material', 'Date'])

# --- 3. เตรียมการทำนายล่วงหน้า (Setup Forecast) ---
# กำหนดช่วงเวลาเป้าหมาย: มิ.ย. - ธ.ค. 2025
TARGET_START_DATE = pd.Timestamp('2025-06-01')
TARGET_END_DATE = pd.Timestamp('2025-12-31')

last_date_in_db = df_daily['Date'].max()
sim_start_date = last_date_in_db + pd.Timedelta(days=1)

print(f"🔮 ช่วงเวลาเป้าหมาย: {TARGET_START_DATE.date()} ถึง {TARGET_END_DATE.date()}")
print(f"⚙️ เริ่มประมวลผลต่อจาก: {last_date_in_db.date()}")

# Features List
features = ['day_of_week', 'is_weekend', 'month_sin', 'x3_Class',
            'Qty_Lag1', 'Qty_Lag7', 'Qty_Lag14', 'Qty_Lag28',
            'Roll_Mean_7D', 'Roll_Mean_30D', 'Roll_Std_30D']
target = 'x1_Quantity'

final_forecasts = []

# --- 4. Main Loop (Per Material) ---
for mat in materials:
    df_mat = df_daily[df_daily['Material'] == mat].copy()
    if len(df_mat) < 60: continue 

    # แปลง Class
    df_mat['x3_Class'] = df_mat['x3_Class'].astype('category')
    current_class = str(df_mat['x3_Class'].iloc[-1]).strip().upper()
    cat_dtype = df_mat['x3_Class'].dtype 

    # ==========================================
    # STEP A: หาค่า Sigma Error (Validation)
    # ==========================================
    val_cutoff = df_mat['Date'].max() - pd.Timedelta(days=90)
    train_val = df_mat[df_mat['Date'] < val_cutoff].copy()
    test_val = df_mat[df_mat['Date'] >= val_cutoff].copy()
    
    def engineer_features(data_subset):
        d = data_subset.copy()
        d['day_of_week'] = d['Date'].dt.dayofweek
        d['is_weekend'] = d['day_of_week'].isin([5, 6]).astype(int)
        d['month'] = d['Date'].dt.month
        d['month_sin'] = np.sin(2 * np.pi * d['month']/12)
        
        d['Qty_Lag1'] = d['x1_Quantity'].shift(1)
        d['Qty_Lag7'] = d['x1_Quantity'].shift(7)
        d['Qty_Lag14'] = d['x1_Quantity'].shift(14)
        d['Qty_Lag28'] = d['x1_Quantity'].shift(28)
        
        d['Roll_Mean_7D'] = d['x1_Quantity'].shift(1).rolling(7).mean()
        d['Roll_Mean_30D'] = d['x1_Quantity'].shift(1).rolling(30).mean()
        d['Roll_Std_30D'] = d['x1_Quantity'].shift(1).rolling(30).std()
        return d

    train_val_feat = engineer_features(train_val).dropna()
    test_val_feat = engineer_features(pd.concat([train_val.iloc[-30:], test_val])).iloc[30:]
    
    sigma_error = 1.0 
    if len(train_val_feat) > 10 and len(test_val_feat) > 0:
        dtrain_val = lgb.Dataset(train_val_feat[features], label=train_val_feat[target], categorical_feature=['x3_Class'])
        params = {
            'objective': 'tweedie', 'metric': 'rmse', 'tweedie_variance_power': 1.1,
            'learning_rate': 0.05, 'max_depth': 5, 'num_leaves': 15, 'verbose': -1, 'seed': 42
        }
        model_val = lgb.train(params, dtrain_val, num_boost_round=200)
        preds_val = model_val.predict(test_val_feat[features])
        residuals = test_val_feat[target] - preds_val
        sigma_error = np.std(residuals)
        
    # ==========================================
    # STEP B: เทรนโมเดลจริง (Full Training)
    # ==========================================
    full_data_feat = engineer_features(df_mat).dropna()
    dtrain_full = lgb.Dataset(full_data_feat[features], label=full_data_feat[target], categorical_feature=['x3_Class'])
    model_full = lgb.train(params, dtrain_full, num_boost_round=300)
    
    # ==========================================
    # STEP C: จำลองอนาคต (Simulation Loop)
    # ==========================================
    sim_df = df_mat[['Date', 'Material', 'x1_Quantity', 'x3_Class']].copy()
    curr_date = sim_start_date 
    forecast_rows = []
    
    while curr_date <= TARGET_END_DATE:
        # 1. สร้าง Features
        day_of_week = curr_date.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        month_sin = np.sin(2 * np.pi * curr_date.month/12)
        
        qty_series = sim_df['x1_Quantity']
        
        # คำนวณ Lag & Rolling (ระวังเรื่อง Index)
        # ใช้ iloc เพื่อดึงข้อมูลท้ายสุดของ sim_df ที่โตขึ้นเรื่อยๆ
        lag1 = qty_series.iloc[-1]
        lag7 = qty_series.iloc[-7] if len(sim_df) >= 7 else 0
        lag14 = qty_series.iloc[-14] if len(sim_df) >= 14 else 0
        lag28 = qty_series.iloc[-28] if len(sim_df) >= 28 else 0
        
        roll_mean_7 = qty_series.iloc[-7:].mean() if len(sim_df) >= 7 else 0
        roll_mean_30 = qty_series.iloc[-30:].mean() if len(sim_df) >= 30 else 0
        roll_std_30 = qty_series.iloc[-30:].std() if len(sim_df) >= 30 else 0
        
        feat_dict = {
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'month_sin': [month_sin],
            'x3_Class': [df_mat['x3_Class'].iloc[-1]],
            'Qty_Lag1': [lag1], 'Qty_Lag7': [lag7], 'Qty_Lag14': [lag14], 'Qty_Lag28': [lag28],
            'Roll_Mean_7D': [roll_mean_7], 'Roll_Mean_30D': [roll_mean_30], 'Roll_Std_30D': [roll_std_30]
        }
        input_df = pd.DataFrame(feat_dict)
        input_df['x3_Class'] = input_df['x3_Class'].astype(cat_dtype)
        
        # 2. ทำนาย (Predict) พร้อม Logic ตัด Noise
        pred_val = model_full.predict(input_df)[0]
        pred_val = max(0, pred_val)
        
        # --- LOGIC แก้ไข: ป้องกันการเบิกทุกวัน (Circuit Breaker) ---
        if is_weekend == 1:
            # กฎ: วันเสาร์-อาทิตย์ ให้เป็น 0 (ถ้าโรงงานทำงานทุกวัน ให้ลบ if นี้ออก)
            pred_qty_int = 0
        else:
            # กฎ: ถ้า AI ไม่มั่นใจ (ค่าน้อยกว่า 0.6) ให้ปัดเป็น 0 เพื่อตัด Noise
            if pred_val < 0.6: 
                pred_qty_int = 0
            else:
                pred_qty_int = int(round(pred_val))
        # -----------------------------------------------------

        # 3. คำนวณ Safety Stock (เก็บไว้ดู)
        selected_z = SERVICE_LEVEL_MAP.get(current_class, DEFAULT_Z)
        ss_val = selected_z * sigma_error
        
        # 4. บันทึกผล (เฉพาะช่วงเวลาเป้าหมาย)
        if curr_date >= TARGET_START_DATE:
            forecast_rows.append({
                'Date': curr_date,
                'Material': mat,
                'Class': current_class,
                'Forecast_AI': pred_qty_int,       
                'Safety_Stock': int(np.ceil(ss_val)) 
            })
        
        # 5. อัปเดตข้อมูลจำลอง
        new_row = pd.DataFrame([{
            'Date': curr_date, 'Material': mat, 
            'x1_Quantity': pred_qty_int, 
            'x3_Class': df_mat['x3_Class'].iloc[-1]
        }])
        sim_df = pd.concat([sim_df, new_row], ignore_index=True)
        
        curr_date += pd.Timedelta(days=1)

    if forecast_rows:
        final_forecasts.extend(forecast_rows)
        print(f"📦 {mat} : Forecast เสร็จสิ้น")

# --- 5. Save Output ---
if final_forecasts:
    res_df = pd.DataFrame(final_forecasts)
    res_df = res_df.sort_values(['Material', 'Date'])
    
    print("-" * 60)
    print(f"💾 กำลังบันทึกไฟล์ผลลัพธ์: {output_filename}")
    
    with pd.ExcelWriter(output_filename) as writer:
        res_df.to_excel(writer, sheet_name='Daily_Forecast', index=False)
        
        res_df['Month'] = res_df['Date'].dt.to_period('M')
        monthly_summary = res_df.groupby(['Material', 'Class', 'Month'])['Forecast_AI'].sum().reset_index()
        monthly_summary.to_excel(writer, sheet_name='Monthly_Summary', index=False)
        
    print(f"✅ เรียบร้อย! เปิดไฟล์ {output_filename} ดูผลลัพธ์ได้เลยครับ")
else:
    print("❌ ไม่สามารถสร้าง Forecast ได้")