# ✅ 필수 라이브러리 임포트
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
import lightgbm as lgb

# ✅ 페이지 설정
st.set_page_config(page_title="대장암 생존 통계 및 예측 모델 분석", layout="wide")
st.title("🧬 대장암 생존 통계 및 예측 모델 분석")

# ✅ 데이터 및 모델 불러오기
df = pd.read_csv("train_selected_complete.csv")
xgb_model = joblib.load("xgb_model (1).pkl")
lgb_model = joblib.load("lgb_model (1).pkl")
input_scaler = joblib.load("input_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# ✅ 탭 구성
tabs = st.tabs([
    "📊 모델 성능 비교",
    "📌 변수 중요도",
    "🧍 유사 환자 통계",
    "📝 생존일수 예측",
    "📈 생존 경향 시나리오"
])

# 🔹 탭 0: 모델 성능 비교
with tabs[0]:
    st.subheader("📊 모델 성능 비교 (XGBoost vs LightGBM)")
    metrics = {
        "MAE": [0.92, 0.95],
        "RMSE": [1.24, 1.27],
        "Pearson": [-0.000, -0.003],
        "Spearman": [-0.018, -0.008],
        "CI": [0.545, 0.497]
    }
    df_result = pd.DataFrame(metrics, index=["XGBoost", "LightGBM"])

    # MAE & RMSE
    st.markdown("### Average Errors")
    fig1, ax1 = plt.subplots()
    bars = df_result[["MAE", "RMSE"]].plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Error")
    ax1.set_title("MAE and RMSE")
    ax1.set_xticks(range(len(df_result.index)))
    ax1.set_xticklabels(df_result.index, rotation=0)
    for i, bar_container in enumerate(ax1.containers):
        for bar in bar_container:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center', va='bottom', fontsize=9)
    st.pyplot(fig1)

    # Correlation
    st.markdown("### Correlation")
    fig2, ax2 = plt.subplots()
    ax2.plot(df_result.index, df_result["Pearson"], marker='o', label='Pearson')
    ax2.plot(df_result.index, df_result["Spearman"], marker='s', label='Spearman')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title("Correlation (r)")
    ax2.set_ylabel("Value")
    for i, val in enumerate(df_result["Pearson"]):
        ax2.text(i, val, f"{val:.3f}", ha='center', va='bottom')
    for i, val in enumerate(df_result["Spearman"]):
        ax2.text(i, val, f"{val:.3f}", ha='center', va='bottom')
    ax2.legend()
    st.pyplot(fig2)

    # CI
    st.markdown("### Concordance Index")
    fig3, ax3 = plt.subplots()
    bars = ax3.barh(df_result.index, df_result["CI"], color="skyblue")
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("CI (higher is better)")
    ax3.set_title("CI Comparison")
    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax3.text(width + 0.02, y, f"{width:.3f}", va='center', fontsize=9)
    st.pyplot(fig3)

# 🔹 탭 1: 변수 중요도
with tabs[1]:
    st.subheader("📌 변수 중요도 ")
    importance_dict = {
        "Weight": 549.0,
        "Height": 485.0,
        "AGE": 399.0,
        "NRAS_MUTATION": 70.0,
        "Drink Type": 60.0,
        "MSI": 56.0,
        "Smoke": 52.0,
        "EGFR": 50.0,
        "KRAS": 49.0,
        "BRAF_MUTATION": 45.0
    }
    imp_df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "F Score"])
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    barplot = sns.barplot(x="F Score", y="Feature", data=imp_df, ax=ax4, color="steelblue")
    ax4.set_title("Important Features")
    for bar in barplot.patches:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax4.text(width + 10, y, f"{width:.1f}", va='center', fontsize=10)
    st.pyplot(fig4)

# 🔹 탭 2: 유사 환자 통계
with tabs[2]:
    st.subheader("🧍 Similar Patient Statistics")
    age_input = st.number_input("Age", min_value=0, max_value=100, value=60)
    chemo_input = st.selectbox("Chemotherapy", [0, 1])
    surgery_input = st.selectbox("Surgery", [0, 1])
    kras_input = st.selectbox("KRAS Mutation", [0, 1])

    similar = df[
        (df["AGE"] // 5 == age_input // 5) &
        (df["Chemo"] == chemo_input) &
        (df["Surgery"] == surgery_input) &
        (df["KRAS"] == kras_input)
    ]
    st.write(f"🔍 Number of similar patients: {len(similar)}")
    if len(similar) > 0:
        st.write(f"📊 Mean Survival Days: {similar['Survival'].mean():.0f} days")
        st.write(f"📊 Median Survival Days: {similar['Survival'].median():.0f} days")
        fig5, ax5 = plt.subplots()
        ax5.hist(similar["Survival"], bins=20)
        ax5.set_title("Survival Days (Similar Patients)")
        ax5.set_xlabel("Survival Days")
        ax5.set_ylabel("Count")
        st.pyplot(fig5)
    else:
        st.info("No similar patients found.")

# 🔹 탭 3: 생존일수 예측
with tabs[3]:
    st.subheader("📝 환자 정보 입력")
    data = {}
    data["나이"] = st.number_input("나이", 0, 120, 60)
    data["키"] = st.number_input("키", 100, 220, 165)
    data["체중"] = st.number_input("체중", 30, 150, 60)
    data["음주종류"] = st.selectbox("음주 종류", [0, 1, 2, 3, 4])
    data["흡연여부"] = st.radio("흡연 여부", [0, 1], format_func=lambda x: "X" if x == 0 else "O")

    for col in ["대장암수술여부", "항암제치료여부", "방사선치료여부"]:
        data[col] = st.radio(col, [0, 1], format_func=lambda x: "X" if x == 0 else "O")

    for gene in ["EGFR", "MSI", "KRAS", "NRAS", "BRAF"]:
        data[gene] = st.selectbox(f"{gene} 돌연변이 여부", [0, 1], format_func=lambda x: "X" if x == 0 else "O")

    # ✅ 병기: One-hot 방식 선택
    stage_options = ["T1", "T2", "T3", "T4", "N1", "N2", "N3", "M1", "Tis"]
    selected_stage = st.selectbox("병기 선택", stage_options)
    for stage in stage_options:
        data[stage] = int(stage == selected_stage)

    # ✅ 조직학적 진단명: One-hot 방식 선택
    hist_features = {
        "adenocarcinoma": "adenocarcinoma",
        "mucinous": "mucinous",
        "signet ring cell": "signet_ring_cell",
        "squamous cell carcinoma": "squamous_cell_carcinoma",
        "Neoplasm malignant": "Neoplasm_malignant",
        "carcinoide tumor": "carcinoide_tumor",
        "Neuroendocrine carcinoma": "Neuroendocrine_carcinoma"
    }
    selected_hist = st.selectbox("조직학적 진단명 선택", list(hist_features.keys()))
    for label, var_name in hist_features.items():
        data[var_name] = int(label == selected_hist)

    model_choice = st.selectbox("모델 선택", ["XGBoost", "LightGBM"])

    if st.button("예측 실행"):
        try:
            input_df = pd.DataFrame([data])

            if model_choice == "XGBoost":
                model = xgb_model
                model_features = model.get_booster().feature_names
            else:
                model = lgb_model
                model_features = model.booster_.feature_name()

            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_features]

            # ✅ 입력 스케일링
            for col in input_scaler.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            scaler_input = input_df[input_scaler.feature_names_in_]
            scaled_array = input_scaler.transform(scaler_input)
            scaled_df = pd.DataFrame(scaled_array, columns=input_scaler.feature_names_in_)

            for col in input_df.columns:
                if col not in scaled_df.columns:
                    scaled_df[col] = input_df[col]

            input_for_model = scaled_df[model_features]

            # ✅ 예측 및 타겟 역변환
            pred_scaled_log = model.predict(input_for_model)
            pred_log = target_scaler.inverse_transform(np.array(pred_scaled_log).reshape(-1, 1))[0][0]
            pred_days = np.expm1(pred_log)

            st.success(f"✅ 예측된 생존일수: {pred_days:.1f}일")

        except Exception as e:
            st.error(f"❌ 예측 중 오류 발생: {e}")

# 🔹 탭 4: 생존 경향 시나리오
with tabs[4]:
    st.subheader("📈 Survival Scenario Trends")

    # 시나리오 선택
    scenario = st.selectbox("Select Scenario", ["No Treatment", "Surgery Only", "Surgery + Chemo"])

    # 시나리오 조건 필터링
    if scenario == "No Treatment":
        subset = df[(df["Surgery"] == 0) & (df["Chemo"] == 0)]
    elif scenario == "Surgery Only":
        subset = df[(df["Surgery"] == 1) & (df["Chemo"] == 0)]
    elif scenario == "Surgery + Chemo":
        subset = df[(df["Surgery"] == 1) & (df["Chemo"] == 1)]

    # 환자 수 출력
    st.write(f"🧪 Number of patients in scenario: {len(subset)}")

    if len(subset) > 0:
        # 평균 및 중앙값 계산
        mean_val = subset["Survival"].mean()
        median_val = subset["Survival"].median()

        st.write(f"📊 Mean Survival Days: {mean_val:.0f} days")
        st.write(f"📊 Median Survival Days: {median_val:.0f} days")

        # 박스플롯 그리기 (tick_labels 사용)
        fig6, ax6 = plt.subplots()
        ax6.boxplot(subset["Survival"], tick_labels=[scenario])
        ax6.set_title("Survival Days by Treatment Scenario")

        # 색상으로 평균/중앙값 텍스트 표시 (선 없이)
        ax6.text(1.1, mean_val, f"Mean: {mean_val:.0f}", color='blue', va='center', fontsize=10)
        ax6.text(1.1, median_val, f"Median: {median_val:.0f}", color='green', va='center', fontsize=10)

        st.pyplot(fig6)
    else:
        st.info("No patients found for this scenario.")
