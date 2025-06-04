import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import joblib
import json

# ✅ 페이지 기본 설정
st.set_page_config(page_title="대장암 환자의 생존일수 예측과 시각화", layout="wide")
st.title("대장암 환자의 생존일수 예측과 시각화")

# ✅ 모델 및 전처리 로드 (경로 수정됨)
xgb_model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")
with open("model/continual_col.json", "r") as f:

# ✅ 탭 구성
탭 = st.tabs(["🧪 생존일수 예측", "📊 모델 성능 비교", "📈 예측 vs 실제", "🔍 변수 중요도"])

# 🔹 탭 1: 생존일수 예측
with 탭[0]:
    st.subheader("📝 환자 정보 입력")

    data = {}
    stage_cols = [col for col in continual_col if col.startswith("병기STAGE(")]

    gene_cols = [
        "면역병리EGFR검사코드/명(EGFR)",
        "분자병리MSI검사결과코드/명(MSI)",
        "분자병리KRASMUTATION_EXON2검사결과코드/명(KRASMUTATION_EXON2)",
        "분자병리KRASMUTATION검사결과코드/명(KRASMUTATION)",
        "분자병리NRASMUTATION검사결과코드/명(NRASMUTATION)",
        "분자병리BRAF_MUTATION검사결과코드/명(BRAF_MUTATION)"
    ]

    for col in continual_col:
        if col == "log_survival":
            continue
        elif col == "진단시연령(AGE)":
            data[col] = st.number_input("진단시 연령(AGE)", min_value=0, max_value=120, value=60)
        elif col == "신장값(Height)":
            data[col] = st.number_input("신장값(Height)", min_value=100, max_value=220, value=165)
        elif col == "체중측정값(Weight)":
            data[col] = st.number_input("체중측정값(Weight)", min_value=30, max_value=150, value=60)
        elif col in stage_cols:
            continue
        elif col == "음주종류(Type of Drink)":
            data[col] = st.selectbox("음주종류(Type of Drink)", options=[0, 1, 2, 3, 4], index=4)
        elif col == "흡연여부(Smoke)":
            data[col] = st.radio("흡연 여부", options=[0, 1], format_func=lambda x: "X" if x == 0 else "O")
        elif col == "대장암 수술 여부(Operation)":
            data[col] = st.radio("대장암 수술 여부", options=[0, 1], format_func=lambda x: "X" if x == 0 else "O")
        elif col == "방사선치료 여부(Radiation Therapy)":
            data[col] = st.radio("방사선 치료 여부", options=[0, 1], format_func=lambda x: "X" if x == 0 else "O")
        elif "여부" in col or "치료" in col:
            data[col] = st.radio(col, options=[0, 1], format_func=lambda x: "X" if x == 0 else "O")
        elif "조직학적진단명 코드 설명" in col:
            data[col] = st.selectbox(col, options=[0, 1])
        elif col in gene_cols:
            data[col] = st.selectbox(
                col,
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "정상",
                    1: "이상 (돌연변이1)",
                    2: "이상 (돌연변이2)",
                    3: "미측정/결측치"
                }.get(x, str(x)),
                index=3
            )
        else:
            data[col] = int(st.checkbox(col, value=False))

    selected_stage = st.selectbox("병기 선택", stage_cols)
    for col in stage_cols:
        data[col] = int(col == selected_stage)

    data["log_survival"] = 0
    input_df = pd.DataFrame([data])[continual_col]

    scaled_input = scaler.transform(input_df)
    df_scaled = pd.DataFrame(scaled_input, columns=continual_col)
    X = df_scaled[xgb_model.feature_names_in_]

    st.write("📌 원본 입력 데이터", input_df)
    st.write("📌 정규화된 입력 데이터", df_scaled)
    st.write("📌 모델 입력 데이터 (X)", X)

    if st.button("🔍 생존일수 예측"):
        log_pred = xgb_model.predict(X)
        st.write("📌 예측된 log 생존일수:", log_pred[0])
        pred_days = np.expm1(log_pred[0])
        st.success(f"✅ 예측된 생존일수: {pred_days:.2f}일")

# 🔹 탭 2: 모델 성능 비교
with 탭[1]:
    st.subheader("📊 모델 성능 비교 (XGBoost vs LightGBM)")
    metrics = {
        "MAE": [0.95, 0.95],
        "RMSE": [1.27, 1.27],
        "Pearson": [-0.000, -0.003],
        "Spearman": [-0.018, -0.008],
        "Concordance Index": [0.543, 0.497]
    }
    df_result = pd.DataFrame(metrics, index=["XGBoost", "LightGBM"])

    st.markdown("### ✅ MAE / RMSE")
    fig1, ax1 = plt.subplots()
    df_result[["MAE", "RMSE"]].plot(kind="bar", ax=ax1)
    ax1.set_ylabel("오차")
    ax1.set_title("모델별 MAE 및 RMSE")
    ax1.set_xticks(range(len(df_result.index)))
    ax1.set_xticklabels(df_result.index, rotation=0)
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("### 🔵 Pearson / Spearman 상관계수")
    fig2, ax2 = plt.subplots()
    ax2.plot(df_result.index, df_result["Pearson"], marker='o', label='Pearson')
    ax2.plot(df_result.index, df_result["Spearman"], marker='s', label='Spearman')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title("상관계수 비교")
    ax2.set_ylabel("상관계수")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("### 📶 Concordance Index")
    fig3, ax3 = plt.subplots()
    ax3.barh(df_result.index, df_result["Concordance Index"], color="skyblue")
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("CI (높을수록 좋음)")
    ax3.set_title("Concordance Index 비교")
    st.pyplot(fig3)

# 🔹 탭 3: 예측 vs 실제
with 탭[2]:
    st.subheader("📈 예측 vs 실제 (XGBoost 기준)")
    from sklearn.model_selection import train_test_split
    df = pd.read_excel("data/train_set.xlsx", sheet_name="Adjusted_syncolorectal_trainset")
    df["log_survival"] = np.log1p(df["암진단후생존일수(Survival period)"])
    df = df.drop(columns=["암진단후생존일수(Survival period)", "사망여부(Death)", "순번(No)"], errors="ignore")
    df = df[continual_col]
    scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(scaled, columns=continual_col)
    X = df_scaled.drop(columns=["log_survival"])
    y = np.expm1(df_scaled["log_survival"])
    xgb_pred = np.expm1(xgb_model.predict(X[xgb_model.feature_names_in_]))

    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=y, y=xgb_pred, ax=ax4)
    ax4.plot([0, max(y)], [0, max(y)], '--', color='gray')
    ax4.set_xlabel("실제 생존일수")
    ax4.set_ylabel("예측 생존일수")
    ax4.set_title("XGBoost 예측 vs 실제")
    st.pyplot(fig4)

# 🔹 탭 4: 변수 중요도
with 탭[3]:
    st.subheader("🔍 변수 중요도 (XGBoost - F score 기준)")
    importance_dict = {
        "체중측정값(Weight)": 549.0,
        "신장값(Height)": 485.0,
        "진단시연령(AGE)": 399.0,
        "분자병리NRASMUTATION검사결과코드/명(NRASMUTATION)": 70.0,
        "음주종류(Type of Drink)": 60.0,
        "분자병리MSI검사결과코드/명(MSI)": 56.0,
        "흡연여부(Smoke)": 52.0,
        "면역병리EGFR검사코드/명(EGFR)": 50.0,
        "분자병리KRASMUTATION검사결과코드/명(KRASMUTATION)": 49.0,
        "분자병리BRAF_MUTATION검사결과코드/명(BRAF_MUTATION)": 45.0
    }

    imp_df = pd.DataFrame(list(importance_dict.items()), columns=["변수", "F score"])
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="F score", y="변수", data=imp_df, ax=ax6, color="steelblue")
    ax6.set_title("XGBoost 변수 중요도 (F score 기준)")
    st.pyplot(fig6)
