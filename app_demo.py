import streamlit as st
import pandas as pd
import joblib

st.title("👗 Sustainable Fashion Assistant (Interactive Demo)")


fabric_model, chem_model, feature_columns = joblib.load("sustainable_models.pkl")


st.sidebar.header("Production Input")
garment_area = st.sidebar.slider("Garment area (m²)", 0.2, 1.5, 0.6)
pieces = st.sidebar.number_input("Number of pieces", value=100, min_value=1)
fabric_width = st.sidebar.selectbox("Fabric roll width (m)", [1.2, 1.4, 1.5])
defect_rate = st.sidebar.slider("Expected defect rate", 0.0, 0.2, 0.02)
fabric_type = st.sidebar.selectbox("Fabric type", ["knit", "woven", "stretch"])
target_shade = st.sidebar.slider("Shade intensity (0-1)", 0.0, 1.0, 0.4)
temperature = st.sidebar.slider("Dye temperature (°C)", 60, 100, 80)

if st.button("Predict"):
 
    X_input = pd.DataFrame(0, index=[0], columns=feature_columns)


    numeric_features = {
        "garment_area_m2": garment_area,
        "pieces": pieces,
        "fabric_width_m": fabric_width,
        "defect_rate": defect_rate,
        "target_shade": target_shade,
        "temperature_C": temperature
    }
    for col, val in numeric_features.items():
        if col in X_input.columns:
            X_input.at[0, col] = val

    fabric_type_col = f"fabric_type_{fabric_type}"
    if fabric_type_col in X_input.columns:
        X_input.at[0, fabric_type_col] = 1

    
    pred_m = fabric_model.predict(X_input)[0]
    pred_l = chem_model.predict(X_input)[0]


    waste_reduction_pct = defect_rate * 100  

    st.subheader("Predicted Material Needs")
    st.write(f"Fabric needed ≈ **{pred_m:.2f} meters**")
    st.write(f"Chemical volume ≈ **{pred_l:.2f} liters**")
    st.write(f"Estimated waste reduction ≈ **{waste_reduction_pct:.1f}%**")


    st.write("---")
    st.subheader("Practical Suggestions")
    st.markdown(
        f"- Round fabric orders to nearest **0.5 m** roll.\n"
        f"- Optimize nesting patterns.\n"
        f"- Reduce defect rate by improving inspection (~{waste_reduction_pct:.1f}% savings).\n"
        f"- For chemical savings: adjust shade or temperature in dye trials."
    )
