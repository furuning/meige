import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import io

# Page Configuration
st.set_page_config(page_title="梅杰综合征亚型预测系统", layout="wide")

# Language selection
language = st.sidebar.selectbox("🌐 语言 / Language", ("中文", "English"))

# Title and Description
if language == "中文":
    st.title("🧠 梅杰综合征 (Meige Syndrome) 亚型预测系统")
    st.markdown("""
    本工具利用基于血液学和代谢特征谱训练的机器学习模型 (XGBoost) 预测梅杰综合征亚型 (MS1, MS2, MS3)。
    基于研究：*揭示梅杰综合征的血液学与代谢异质性：一种用于客观分型的数据驱动聚类与机器学习方法*
    """)
else:
    st.title("🧠 Meige Syndrome Subtype Prediction System")
    st.markdown("""
    This tool uses a machine learning model (XGBoost) trained on hematological and metabolic profiles to predict Meige syndrome subtypes (MS1, MS2, MS3).
    Based on research: *Unveiling hematological and metabolic heterogeneity of Meige syndrome: A data-driven clustering and machine learning approach for objective subtyping*
    """)

# Load the Meige Syndrome model directly from current directory
MODEL_FILE = "XGBoost.pkl"

if not os.path.exists(MODEL_FILE):
    if language == "中文":
        st.error(f"模型文件 '{MODEL_FILE}' 未找到！请确保训练好的模型文件与当前脚本在同一目录下。")
    else:
        st.error(f"Model file '{MODEL_FILE}' not found! Please ensure the trained model file is in the same directory as this script.")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    if language == "中文":
        st.error(f"加载模型时出错：{e}")
    else:
        st.error(f"Error loading model: {e}")
    st.stop()

# Define the 23 selected feature names exactly as used in training
feature_names = [
    "LDL-C", "INR", "UA", "MCHC", "TP", "AT-III", "TG", "ALB", "ALP", "FIB", 
    "Cr", "PS", "HGB", "Ca", "HDL-C", "NEU", "sdLDL-C", "LANR", "PC", "PLT", 
    "DBP", "BMI", "Sex"
]

# Create two columns for input organization
col1, col2 = st.columns(2)

if language == "中文":
    with col1:
        st.subheader("🩸 脂质与代谢指标")
        ldl_c = st.number_input("LDL-C (mmol/L)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        tg = st.number_input("TG (mmol/L)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        hdl_c = st.number_input("HDL-C (mmol/L)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
        sdldl_c = st.number_input("sdLDL-C (mmol/L)", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
        ua = st.number_input("尿酸 - UA (μmol/L)", min_value=100, max_value=800, value=350, step=1)
        ca = st.number_input("钙 - Ca (mmol/L)", min_value=1.5, max_value=3.5, value=2.3, step=0.1)
        alb = st.number_input("白蛋白 - ALB (g/L)", min_value=20, max_value=60, value=40, step=1)
        tp = st.number_input("总蛋白 - TP (g/L)", min_value=40, max_value=90, value=70, step=1)
        alp = st.number_input("碱性磷酸酶 - ALP (U/L)", min_value=20, max_value=300, value=80, step=1)
        cr = st.number_input("肌酐 - Cr (μmol/L)", min_value=30, max_value=500, value=80, step=1)

    with col2:
        st.subheader("🩹 凝血与血液学指标")
        inr = st.number_input("INR", min_value=0.5, max_value=5.0, value=1.0, step=0.05)
        fib = st.number_input("纤维蛋白原 - FIB (g/L)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
        at_iii = st.number_input("抗凝血酶III - AT-III (%)", min_value=50, max_value=150, value=100, step=1)
        ps = st.number_input("蛋白S - PS (%)", min_value=50, max_value=150, value=100, step=1)
        pc = st.number_input("蛋白C - PC (%)", min_value=50, max_value=150, value=100, step=1)
        pl_t = st.number_input("血小板计数 - PLT (×10⁹/L)", min_value=50, max_value=500, value=250, step=1)
        hgb = st.number_input("血红蛋白 - HGB (g/L)", min_value=60, max_value=200, value=140, step=1)
        mchc = st.number_input("平均红细胞血红蛋白浓度 - MCHC (g/L)", min_value=250, max_value=400, value=330, step=1)
        neu = st.number_input("中性粒细胞计数 - NEU (×10⁹/L)", min_value=1.0, max_value=20.0, value=4.0, step=0.1)
        lanr = st.number_input("LANR (比值/指数)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        
        st.subheader("👤 人口统计学特征")
        dbp = st.number_input("舒张压 - DBP (mmHg)", min_value=40, max_value=120, value=80, step=1)
        bmi = st.number_input("体重指数 - BMI", min_value=15.0, max_value=45.0, value=24.0, step=0.1)
        sex = st.selectbox("性别", options=["男", "女"])

    # Map Sex to numerical value (IMPORTANT: Verify encoding from training)
    # Assuming: Male=0, Female=1 (adjust based on your training data)
    sex_encoded = 1 if sex == "女" else 0 

else:  # English version
    with col1:
        st.subheader("🩸 Lipid & Metabolic Markers")
        ldl_c = st.number_input("LDL-C (mmol/L)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        tg = st.number_input("TG (mmol/L)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        hdl_c = st.number_input("HDL-C (mmol/L)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
        sdldl_c = st.number_input("sdLDL-C (mmol/L)", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
        ua = st.number_input("Uric Acid - UA (μmol/L)", min_value=100, max_value=800, value=350, step=1)
        ca = st.number_input("Calcium - Ca (mmol/L)", min_value=1.5, max_value=3.5, value=2.3, step=0.1)
        alb = st.number_input("Albumin - ALB (g/L)", min_value=20, max_value=60, value=40, step=1)
        tp = st.number_input("Total Protein - TP (g/L)", min_value=40, max_value=90, value=70, step=1)
        alp = st.number_input("Alkaline Phosphatase - ALP (U/L)", min_value=20, max_value=300, value=80, step=1)
        cr = st.number_input("Creatinine - Cr (μmol/L)", min_value=30, max_value=500, value=80, step=1)

    with col2:
        st.subheader("🩹 Coagulation & Hematology Markers")
        inr = st.number_input("INR", min_value=0.5, max_value=5.0, value=1.0, step=0.05)
        fib = st.number_input("Fibrinogen - FIB (g/L)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
        at_iii = st.number_input("Antithrombin III - AT-III (%)", min_value=50, max_value=150, value=100, step=1)
        ps = st.number_input("Protein S - PS (%)", min_value=50, max_value=150, value=100, step=1)
        pc = st.number_input("Protein C - PC (%)", min_value=50, max_value=150, value=100, step=1)
        pl_t = st.number_input("Platelet Count - PLT (×10⁹/L)", min_value=50, max_value=500, value=250, step=1)
        hgb = st.number_input("Hemoglobin - HGB (g/L)", min_value=60, max_value=200, value=140, step=1)
        mchc = st.number_input("Mean Corpuscular Hemoglobin Concentration - MCHC (g/L)", min_value=250, max_value=400, value=330, step=1)
        neu = st.number_input("Neutrophil Count - NEU (×10⁹/L)", min_value=1.0, max_value=20.0, value=4.0, step=0.1)
        lanr = st.number_input("LANR (Ratio/Index)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        
        st.subheader("👤 Demographics")
        dbp = st.number_input("Diastolic Blood Pressure - DBP (mmHg)", min_value=40, max_value=120, value=80, step=1)
        bmi = st.number_input("Body Mass Index - BMI", min_value=15.0, max_value=45.0, value=24.0, step=0.1)
        sex = st.selectbox("Sex", options=["Male", "Female"])

    # Map Sex to numerical value (IMPORTANT: Verify encoding from training)
    # Assuming: Male=0, Female=1 (adjust based on your training data)
    sex_encoded = 1 if sex == "Female" else 0 

# Collect input values into a list matching the feature_names order exactly
feature_values = [
    ldl_c, inr, ua, mchc, tp, at_iii, tg, alb, alp, fib, 
    cr, ps, hgb, ca, hdl_c, neu, sdldl_c, lanr, pc, pl_t, 
    dbp, bmi, sex_encoded
]

# Convert to DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if language == "中文":
    button_label = "🔍 预测 MS 亚型"
    divider_text = "预测结果"
    subtype_label = "预测亚型"
    confidence_label = "置信度"
    clinical_interpretation = "📋 临床解读与建议"
    shap_analysis = "🔬 模型可解释性分析 (SHAP)"
    waterfall_caption = "SHAP 瀑布图 - 特征贡献度分析"
    feature_importance = "📊 特征重要性排序 (Top 10)"
    top_features = "Top 10 特征重要性"
    disclaimer = "注：红色条形表示增加该亚型概率；蓝色条形表示降低该亚型概率。"
    error_prediction = "预测或解释过程中发生错误："
    debug_info = "调试信息："
    input_shape = "输入特征形状："
    feature_names_col = "特征列名："
    input_dtype = "输入数据类型："
    warning_shap = "SHAP 可视化暂时无法显示："
    fallback_message = "但预测结果仍然有效。您可以查看下方的特征贡献度表格。"
    influence_direction = "影响方向"
    increase_prob = "↑ 增加概率"
    decrease_prob = "↓ 降低概率"
    disclaimer_footer = "**免责声明：** 本工具仅用于研究和教育目的，不能替代专业医疗诊断或治疗。临床决策请务必咨询合格的神经科医生。"

else:  # English version
    button_label = "🔍 Predict MS Subtype"
    divider_text = "Prediction Results"
    subtype_label = "Predicted Subtype"
    confidence_label = "Confidence"
    clinical_interpretation = "📋 Clinical Interpretation & Recommendations"
    shap_analysis = "🔬 Model Interpretability Analysis (SHAP)"
    waterfall_caption = "SHAP Waterfall Plot - Feature Contribution Analysis"
    feature_importance = "📊 Feature Importance Ranking (Top 10)"
    top_features = "Top 10 Feature Importance"
    disclaimer = "Note: Red bars indicate increased probability of this subtype; blue bars indicate decreased probability."
    error_prediction = "Error occurred during prediction or interpretation: "
    debug_info = "**Debug Info:**"
    input_shape = "Input feature shape: "
    feature_names_col = "Feature column names: "
    input_dtype = "Input data types: "
    warning_shap = "SHAP visualization temporarily unavailable: "
    fallback_message = "However, prediction results remain valid. You can view the feature contribution table below."
    influence_direction = "Influence Direction"
    increase_prob = "↑ Increase Probability"
    decrease_prob = "↓ Decrease Probability"
    disclaimer_footer = "**Disclaimer:** This tool is for research and educational purposes only and cannot replace professional medical diagnosis or treatment. For clinical decisions, please consult qualified neurologists."

if st.button(button_label, type="primary"):
    try:
        # Prediction
        predicted_class_idx = model.predict(features_df)[0]
        predicted_proba = model.predict_proba(features_df)[0]
        
        # Map index to Class Name (IMPORTANT: Verify class mapping from training)
        class_map = {0: "MS1", 1: "MS2", 2: "MS3"}
        predicted_class_name = class_map.get(predicted_class_idx, f"Class {predicted_class_idx}")
        
        # Calculate probability for use in clinical interpretations
        probability = predicted_proba[predicted_class_idx] * 100
        
        # Display Results
        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label=subtype_label, value=predicted_class_name)
            st.write(f"**{confidence_label}：** {predicted_proba[predicted_class_idx]*100:.2f}%")
            
            # Probability Bar Chart
            prob_df = pd.DataFrame({
                'Subtype' if language == "English" else '亚型': ['MS1', 'MS2', 'MS3'],
                'Probability' if language == "English" else '概率': predicted_proba
            })
            st.bar_chart(prob_df.set_index('Subtype' if language == "English" else '亚型'), color="#4CAF50")

        with col_res2:
            st.subheader(clinical_interpretation)
            
            advice_text = ""
            if predicted_class_name == "MS1":
                if language == "中文":
                    advice_text = (
                        f"**预测亚型：MS1（凝血 - 脂质轴）**\n\n"
                        f"模型识别该患者属于 **MS1** 亚型的概率为 **{probability:.1f}%**。\n\n"
                        "**主要特征：** 该亚型主要由脂质代谢和凝血因子驱动。\n"
                        "- **关键驱动因素：** LDL-C、INR、血小板 (PLT)、蛋白C (PC)、纤维蛋白原 (FIB)\n"
                        "- **临床洞察：** 提示可能与高凝状态和血脂异常相关的表型。"
                        "建议评估血管风险因素和微循环状态。"
                    )
                else:
                    advice_text = (
                        f"**Predicted Subtype: MS1 (Coagulation - Lipid Axis)**\n\n"
                        f"The model identifies this patient as belonging to **MS1** subtype with probability **{probability:.1f}%**.\n\n"
                        "**Main Characteristics:** This subtype is primarily driven by lipid metabolism and coagulation factors.\n"
                        "- **Key Drivers:** LDL-C, INR, Platelets (PLT), Protein C (PC), Fibrinogen (FIB)\n"
                        "- **Clinical Insight:** Suggests a phenotype possibly related to hypercoagulable state and dyslipidemia. "
                        "Recommend assessment of vascular risk factors and microcirculation status."
                    )
            elif predicted_class_name == "MS2":
                if language == "中文":
                    advice_text = (
                        f"**预测亚型：MS2（肾 - 性别轴）**\n\n"
                        f"模型识别该患者属于 **MS2** 亚型的概率为 **{probability:.1f}%**。\n\n"
                        "**主要特征：** 以肾功能标志物和性别特异性差异为特征。\n"
                        "- **关键驱动因素：** HDL-C、血红蛋白 (HGB)、性别、INR、肌酐 (Cr)、尿酸 (UA)\n"
                        "- **临床洞察：** 提示代谢 - 内分泌机制。"
                        "需关注神经活性代谢物的肾脏清除及潜在的激素相互作用。"
                    )
                else:
                    advice_text = (
                        f"**Predicted Subtype: MS2 (Renal - Gender Axis)**\n\n"
                        f"The model identifies this patient as belonging to **MS2** subtype with probability **{probability:.1f}%**.\n\n"
                        "**Main Characteristics:** Characterized by renal function markers and gender-specific differences.\n"
                        "- **Key Drivers:** HDL-C, Hemoglobin (HGB), Sex, INR, Creatinine (Cr), Uric Acid (UA)\n"
                        "- **Clinical Insight:** Suggests metabolic-endocrine mechanisms. "
                        "Attention should be paid to renal clearance of neuroactive metabolites and potential hormonal interactions."
                    )
            elif predicted_class_name == "MS3":
                if language == "中文":
                    advice_text = (
                        f"**预测亚型：MS3（炎症 - 代谢轴）**\n\n"
                        f"模型识别该患者属于 **MS3** 亚型的概率为 **{probability:.1f}%**。\n\n"
                        "**主要特征：** 由炎症和嘌呤代谢标志物定义。\n"
                        "- **关键驱动因素：** 血红蛋白 (HGB)、尿酸 (UA)、sdLDL-C、中性粒细胞 (NEU)、LANR\n"
                        "- **临床洞察：** 提示系统性炎症状态或代谢应激。"
                        "抗炎策略或代谢调节可能是相关考虑方向。"
                    )
                else:
                    advice_text = (
                        f"**Predicted Subtype: MS3 (Inflammation - Metabolism Axis)**\n\n"
                        f"The model identifies this patient as belonging to **MS3** subtype with probability **{probability:.1f}%**.\n\n"
                        "**Main Characteristics:** Defined by inflammatory and purine metabolism markers.\n"
                        "- **Key Drivers:** Hemoglobin (HGB), Uric Acid (UA), sdLDL-C, Neutrophils (NEU), LANR\n"
                        "- **Clinical Insight:** Indicates systemic inflammatory state or metabolic stress. "
                        "Anti-inflammatory strategies or metabolic regulation might be relevant considerations."
                    )
            
            st.info(advice_text)

        # SHAP Explanation - FIXED VERSION
        st.divider()
        st.subheader(shap_analysis)
        if language == "中文":
            st.write(f"以下瀑布图展示了各特征对 **{predicted_class_name}** 亚型预测的贡献度。")
        else:
            st.write(f"The following waterfall plot shows the contribution of each feature to the **{predicted_class_name}** subtype prediction.")

        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(features_df)
            
            # Get SHAP values for the predicted class
            # shap_values object shape: (nsamples, nfeatures, nclasses)
            target_shap = shap_values[0, :, predicted_class_idx]
            
            # Method 1: Save to buffer and display (compatible with most SHAP versions)
            buffer = io.BytesIO()
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(target_shap, show=False)
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            buffer.seek(0)
            st.image(buffer, caption=waterfall_caption)
            
            # Method 2: Also display feature importance bar chart as alternative
            st.subheader(feature_importance)
            
            # Get absolute SHAP values for ranking
            shap_abs = np.abs(target_shap.values)
            feature_importance_df = pd.DataFrame({
                'Feature' if language == "English" else '特征': feature_names,
                'Importance' if language == "English" else '重要性': shap_abs
            }).sort_values('Importance' if language == "English" else '重要性', ascending=False).head(10)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.barh(feature_importance_df['Feature' if language == "English" else '特征'], 
                     feature_importance_df['Importance' if language == "English" else '重要性'], 
                     color='steelblue')
            ax2.set_xlabel('Mean |SHAP Value|' if language == "English" else '平均 |SHAP 值|')
            ax2.set_title(top_features if language == "English" else 'Top 10 特征重要性')
            ax2.invert_yaxis()
            st.pyplot(fig2)
            
        except Exception as shap_error:
            st.warning(f"{warning_shap}{shap_error}")
            st.write(fallback_message)
            
            # Fallback: Display SHAP values as table
            target_shap = shap_values[0, :, predicted_class_idx]
            shap_df = pd.DataFrame({
                'Feature' if language == "English" else '特征': feature_names,
                'SHAP Value' if language == "English" else 'SHAP值': target_shap.values,
                'Feature Value' if language == "English" else '特征值': features_df.iloc[0].values
            })
            shap_df[influence_direction if language == "English" else '影响方向'] = shap_df['SHAP Value' if language == "English" else 'SHAP值'].apply(
                lambda x: increase_prob if x > 0 else decrease_prob
            )
            st.dataframe(shap_df.sort_values('SHAP Value' if language == "English" else 'SHAP值', ascending=False, key=abs))

        st.caption(disclaimer)

    except Exception as e:
        st.error(f"{error_prediction}{e}")
        if language == "中文":
            st.write("请检查输入数据类型是否与模型预期匹配。")
        else:
            st.write("Please check if the input data types match what the model expects.")

        # Debug info
        st.write(debug_info)
        st.write(f"{input_shape}{features_df.shape}")
        st.write(f"{feature_names_col}{list(features_df.columns)}")
        st.write(f"{input_dtype}{features_df.dtypes}")

# Footer
st.markdown("---")
st.markdown(disclaimer_footer)