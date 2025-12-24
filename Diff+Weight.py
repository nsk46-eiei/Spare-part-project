import pandas as pd
import numpy as np
import datetime
import random
import warnings

# ตรวจสอบ Library
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("❌ ไม่พบ sklearn! กรุณาพิมพ์ 'pip install scikit-learn' ใน Terminal")
    exit()

warnings.filterwarnings('ignore')

# --- ⚙️ ตั้งค่าความเข้มข้นของข้อมูล (ปรับตรงนี้ได้เลย) ---
# 1.0 = เท่าของจริง (น้อย), 3.0 = เยอะกว่าเดิม 3 เท่า, 5.0 = เยอะกว่าเดิม 5 เท่า
# แนะนำ 3.0 - 5.0 สำหรับทำ ML ครับ
DATA_MULTIPLIER = 4.0 

# --- 1. ตั้งค่าและเตรียมข้อมูล ---
file_name = 'data.xlsx'
WINDOW_SIZE = 15   
STEPS = 30         

print(f"🚀 เริ่มระบบ Diffusion สำหรับ ML (Target Multiplier: x{DATA_MULTIPLIER})...")

try:
    # อ่านไฟล์
    df = pd.read_excel(file_name)
    df.columns = [str(c).strip() for c in df.columns]
    
    date_col = 'requisition date'
    qty_col = '#Requisition'
    if date_col not in df.columns: date_col = df.columns[6]
    if qty_col not in df.columns: qty_col = df.columns[7]

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').abs()
    
    # เก็บข้อมูลสินค้าและความถี่ (Weight)
    print("📦 วิเคราะห์ความสัมพันธ์ของสินค้า...")
    detail_columns = [c for c in df.columns if c not in [date_col, qty_col, 'Data Type']]
    unique_df = df[detail_columns].drop_duplicates()
    population = unique_df.to_dict('records')
    
    id_col = df.columns[0]
    freq_map = df[id_col].value_counts().to_dict()
    weights = [freq_map.get(item[id_col], 1) for item in population]

    # เตรียมข้อมูลสอน AI
    # ใช้ .size() เพื่อนับ Transaction ต่อวัน (Pattern การเกิดงาน)
    df_daily = df.groupby(date_col).size().resample('D').sum().fillna(0) 
    data_values = df_daily.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_values).flatten()

    # --- 2. สร้าง Training Data ---
    X_train_seq = []
    for i in range(len(data_scaled) - WINDOW_SIZE):
        X_train_seq.append(data_scaled[i : i + WINDOW_SIZE])
    X_train_seq = np.array(X_train_seq)

    # --- 3. Diffusion Parameters ---
    betas = np.linspace(0.0001, 0.02, STEPS)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    # --- 4. สอน Model ---
    print("🤖 AI กำลังเรียนรู้ Pattern การเบิกจ่าย (Training)...")
    X_input = []
    y_target = []

    # เพิ่มรอบการสอน (Epochs) เพื่อให้จำ Pattern แม่นๆ
    for _ in range(10): 
        for i in range(len(X_train_seq)):
            t = np.random.randint(0, STEPS)
            noise = np.random.randn(WINDOW_SIZE)
            clean_seq = X_train_seq[i]
            noisy_seq = (sqrt_alphas_cumprod[t] * clean_seq) + \
                        (sqrt_one_minus_alphas_cumprod[t] * noise)
            
            input_vector = np.append(noisy_seq, t / STEPS)
            X_input.append(input_vector)
            y_target.append(noise)

    model = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=42)
    model.fit(X_input, y_target)
    print("✅ โมเดลเรียนรู้เสร็จแล้ว!")

    # --- 5. สร้างข้อมูลย้อนหลัง (Generation) ---
    first_date_real = df[date_col].min()
    start_back_date = pd.Timestamp('2020-01-01')
    days_needed = (first_date_real - start_back_date).days
    
    print(f"✨ กำลังสร้างข้อมูลย้อนหลังแบบ High-Density สำหรับ ML ({days_needed} วัน)...")
    
    num_segments = (days_needed // WINDOW_SIZE) + 1
    synthetic_counts_seq = []

    for seg in range(num_segments):
        current_seq = np.random.randn(WINDOW_SIZE)
        for t in reversed(range(STEPS)):
            input_vector = np.append(current_seq, t / STEPS).reshape(1, -1)
            predicted_noise = model.predict(input_vector)[0]
            
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]
            noise_z = np.random.randn(WINDOW_SIZE) if t > 0 else 0
            
            current_seq = (1 / np.sqrt(alpha)) * (current_seq - ((1 - alpha) / (np.sqrt(1 - alpha_cumprod))) * predicted_noise) + (np.sqrt(beta) * noise_z)
        synthetic_counts_seq.extend(current_seq)

    synthetic_counts_seq = synthetic_counts_seq[:days_needed]
    synthetic_counts_scaled = np.array(synthetic_counts_seq).reshape(-1, 1)
    synthetic_counts_raw = scaler.inverse_transform(synthetic_counts_scaled).flatten()
    
    # --- [จุดสำคัญ] Calibration แบบเร่งยอด (Augmentation) ---
    # คำนวณยอดเป้าหมายแบบ "ทวีคูณ" (Multiplier)
    avg_rows_per_day = len(df) / ((df[date_col].max() - df[date_col].min()).days + 1)
    target_total = int(avg_rows_per_day * days_needed * DATA_MULTIPLIER)
    
    current_sum = np.sum(np.abs(synthetic_counts_raw))
    scaling_factor = target_total / current_sum if current_sum != 0 else 1
    
    # ปรับค่าขึ้น (Upscaling)
    synthetic_counts = np.floor(np.abs(synthetic_counts_raw) * scaling_factor).astype(int)
    
    # เติม Randomness เล็กน้อยเพื่อให้ข้อมูลไม่นิ่งเกินไป (ดีต่อ ML)
    synthetic_counts = synthetic_counts + np.random.choice([0, 1], size=len(synthetic_counts), p=[0.7, 0.3])

    print(f"📊 สรุปปริมาณข้อมูลที่สร้าง: {np.sum(synthetic_counts)} รายการ (จากเป้าหมาย x{DATA_MULTIPLIER})")
    print(f"   (ปริมาณนี้เหมาะสำหรับการนำไปทำ ML Training ครับ)")

    # --- 6. แปลงเป็นรายการสินค้า ---
    synthetic_data = []
    current_date = start_back_date
    avg_qty_per_job = df[qty_col].mean()

    for daily_job_count in synthetic_counts:
        if daily_job_count > 0:
            # ใช้ Weighted Random Choice เหมือนเดิม (รักษาความจริงว่าสินค้าไหนฮิต)
            selected_items = random.choices(population, weights=weights, k=daily_job_count)
            for item in selected_items:
                new_row = item.copy()
                new_row[date_col] = current_date
                
                # Variation ของจำนวนชิ้น (ให้ ML เรียนรู้ความหลากหลาย)
                noise_qty = np.random.uniform(0.5, 1.8) # แกว่งกว้างขึ้นนิดนึง
                sim_qty = max(1, int(avg_qty_per_job * noise_qty))
                new_row[qty_col] = sim_qty
                
                new_row['Data Type'] = 'Synthetic (ML Augmented)'
                synthetic_data.append(new_row)
        
        current_date += datetime.timedelta(days=1)

    # --- 7. บันทึก ---
    if synthetic_data:
        df_past = pd.DataFrame(synthetic_data)
        df['Data Type'] = 'Actual'
        df_final = pd.concat([df_past, df], ignore_index=True)
        
        output_filename = "ML_Ready_Sparepart_Data.xlsx"
        df_final.sort_values(by=date_col).to_excel(output_filename, index=False)
        print(f"\n🎉 สำเร็จ! ไฟล์สำหรับทำ ML พร้อมแล้วครับ: {output_filename}")
        print(f"Total Rows: {len(df_final)}")
    else:
        print("❌ ไม่มีการสร้างข้อมูล (ลองรันใหม่อีกครั้ง)")

except Exception as e:
    print(f"\n❌ Error: {e}")