import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 페이지 기본 설정
st.set_page_config(page_title="대장암 생존 통계 및 모델 해석", layout="wide")
st.title("🧬 대장암 생존 통계 및 예측 모델 분석")

# ✅ 데이터 불러오기 (train_selected_complete.csv가 있어야 함)
df = pd.read_csv("train_selected_complete.csv")

# ✅ 탭 구성
tabs = st.tabs([
    "📊 모델 성능 비교",
    "📌 변수 중요도",
    "🧍 유사 환자 통계",
    "📈 생존 경향 시나리오",
    "🧬 주요 유전자별 생존일수 비교"
])

# 🔹 탭 1: 모델 성능 비교
with tabs[0]:
    st.subheader("📊 모델 성능 비교 (XGBoost vs LightGBM)")
    metrics = {
        "MAE": [0.95, 0.95],
        "RMSE": [1.27, 1.27],
        "Pearson": [-0.000, -0.003],
        "Spearman": [-0.018, -0.008],
        "CI": [0.543, 0.497]
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

# 🔹 탭 2: 변수 중요도 (예시)
with tabs[1]:
    st.subheader("📌 변수 중요도 (예시 값)")
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

# 🔹 탭 3: 유사 환자 통계
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

# 🔹 탭 4: 생존 경향 시나리오
with tabs[3]:
    st.subheader("📈 Survival Scenario Trends")
    scenario = st.selectbox("Select Scenario", ["No Treatment", "Surgery Only", "Surgery + Chemo"])
    if scenario == "No Treatment":
        subset = df[(df["Surgery"] == 0) & (df["Chemo"] == 0)]
    elif scenario == "Surgery Only":
        subset = df[(df["Surgery"] == 1) & (df["Chemo"] == 0)]
    elif scenario == "Surgery + Chemo":
        subset = df[(df["Surgery"] == 1) & (df["Chemo"] == 1)]

    st.write(f"🧪 Number of patients in scenario: {len(subset)}")
    if len(subset) > 0:
        st.write(f"📊 Mean Survival Days: {subset['Survival'].mean():.0f} days")
        st.write(f"📊 Median Survival Days: {subset['Survival'].median():.0f} days")
        fig6, ax6 = plt.subplots()
        ax6.boxplot(subset["Survival"], labels=[scenario])
        ax6.set_title("Survival Days by Treatment Scenario")
        st.pyplot(fig6)
    else:
        st.info("No patients found for this scenario.")

# 🔹 탭 5: 유전자별 생존일수 비교
with tabs[4]:
    st.subheader("🧬 주요 유전자별 생존일수 비교 (KRAS, BRAF, EGFR, MSI)")

    gene_map = {
        "KRAS": "KRAS",
        "BRAF": "BRAF_MUTATION",
        "EGFR": "EGFR",
        "MSI": "MSI"
    }

    for gene_name, gene_col in gene_map.items():
        st.markdown(f"### 🔹 {gene_name} 돌연변이 상태별 생존일수 분포")

        if gene_col not in df.columns:
            st.warning(f"⚠️ {gene_name} 컬럼이 존재하지 않습니다.")
            continue

        df_gene = df[df[gene_col] != 99]
        groups = sorted(df_gene[gene_col].unique())
        survival_data = [df_gene[df_gene[gene_col] == g]["Survival"] for g in groups]
        group_labels = [f"{gene_name}={g}" for g in groups]

        for g, surv in zip(groups, survival_data):
            st.write(f"• {gene_name} = {g} → 환자 수: {len(surv)}명, 평균 생존일수: {surv.mean():.0f}일, 중앙값: {surv.median():.0f}일")

        fig, ax = plt.subplots()
        ax.boxplot(survival_data, labels=group_labels)
        ax.set_title(f"Survival Days by {gene_name} Status")
        ax.set_ylabel("Survival Days")
        st.pyplot(fig)

        st.markdown("---")
