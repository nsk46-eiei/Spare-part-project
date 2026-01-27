import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# 1. โหลดข้อมูล
file_path = 'Data_Final_With_TimeBetween.xlsx' # ตรวจสอบชื่อไฟล์ให้ถูกต้อง
df = pd.read_excel(file_path)

# 2. Preprocessing
categorical_cols = ['Material', 'type of material', 'Class']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# 3. กำหนด Features (เพิ่มชุดข้อมูลที่สำคัญ)
features = [
    'Spare part', 'Material', 'type of material', 'Class', 'Lead time',
    'Unit/Price', 'Year', 'Month', 'Day', 'Weekday', 'Quarter',
    'Is_Month_End', 'Is_Weekend', 'Days_Since_Last_Req', 'Prev_Req_Qty',
    'Order_Sequence', 'Cumulative_Qty', 'Rolling_Avg_Qty_3',
    'Rolling_Max_Qty_3', 'Class_Score', 'Time_Between_Requisition', 'Avg_Time_Between_Req'
]
target = '#Requisition'

# 4. แบ่งข้อมูล Train (Scenario อื่นๆ) และ Test (Actual)
train_df = df[df['Scenario'] != 'Actual'].copy()
test_df = df[df['Scenario'] == 'Actual'].copy()

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# 5. กำหนดช่วงการจูนพารามิเตอร์ (Grid Search)
param_grids = {
    'LightGBM': {
        'n_estimators': [100, 500, 1000,2000],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20]
    },
    'Random Forest': {
        'n_estimators': [100, 300,1000],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [1, 2, 4]
    }
}

# 6. เริ่มการ Tuning และหาโมเดลที่ดีที่สุด
models = {
    'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

results_preds = test_df[['Material', 'requisition date', target]].copy()
results_preds.rename(columns={target: 'Actual'}, inplace=True)
metrics_list = []

print("--- Starting Hyperparameter Tuning ---")

for name, model in models.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # ใช้โมเดลที่ดีที่สุดจากการจูน
    best_model = grid.best_estimator_
    print(f"Best Params for {name}: {grid.best_params_}")
    
    # พยากรณ์
    preds = best_model.predict(X_test)
    results_preds[f'Predicted_{name}'] = preds

    # คำนวณ Metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    metrics_list.append({
        'Model': name,
        'MAE': mae,
        'RMAE': mae / np.mean(y_test) if np.mean(y_test) != 0 else 0,
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_test, preds),
        'Best_Params': str(grid.best_params_)
    })

# แปลง Material กลับเป็นชื่อจริง
results_preds['Material'] = le_dict['Material'].inverse_transform(results_preds['Material'])

# 7. บันทึกผลลง Excel
metrics_df = pd.DataFrame(metrics_list)
output_file = 'ML_Tuned_Results2.xlsx'

with pd.ExcelWriter(output_file) as writer:
    results_preds.to_excel(writer, sheet_name='Predictions', index=False)
    metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)

print("-" * 30)
print(f"การจูนเสร็จสมบูรณ์! ผลลัพธ์ถูกบันทึกที่: {output_file}")
print(metrics_df[['Model', 'MAE', 'RMAE', 'MSE', 'RMSE', 'R2']])
