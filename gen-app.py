import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from lime import lime_tabular
from datetime import datetime
import os
np.random.seed(42)

st.set_page_config(layout="wide")

st.title("Generic Classification App")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    target_column = 'Target'
    feature_columns = [col for col in data.columns if col != target_column]
    feature_columns =[col for col in feature_columns if "ID" not in col and "id" not in col]
    id_column = [col for col in data.columns if "ID" in col or "id" in col][0]

    X_train, X_test, Y_train, Y_test = train_test_split(
        data[feature_columns], data[target_column], train_size=0.8, random_state=123
    )

    model = RandomForestClassifier(n_estimators=100, random_state=100)
    model.fit(X_train, Y_train)

    Y_test_preds = model.predict(X_test)
    data['Model_preds'] = model.predict(data[feature_columns])

    st.sidebar.markdown("Classification using RandomForestClassifier ")
    menu = ["Data Preview", "Model Performance", "Model Evaluation & Feedback"]
    choice = st.sidebar.radio("Select a page", menu)

    if choice == "Data Preview":
        st.header("Dataset Preview")
        st.write(data)

    elif choice == "Model Performance":
        st.header("Confusion Matrix | Feature Importances")
        col1, col2 = st.columns(2)
        with col1:
            conf_mat_fig = plt.figure(figsize=(6, 6))
            ax1 = conf_mat_fig.add_subplot(111)
            skplt.metrics.plot_confusion_matrix(Y_test, Y_test_preds, ax=ax1, normalize=True)
            st.pyplot(conf_mat_fig, use_container_width=True)

        with col2:
            feat_imp_fig = plt.figure(figsize=(6, 6))
            ax1 = feat_imp_fig.add_subplot(111)
            skplt.estimators.plot_feature_importances(model, feature_names=feature_columns, ax=ax1, x_tick_rotation=90)
            st.pyplot(feat_imp_fig, use_container_width=True)

        st.divider()
        st.header("Classification Report")
        st.code(classification_report(Y_test, Y_test_preds))

    elif choice == "Model Evaluation & Feedback":
        st.header("Model Prediction and Explanation")

        selected_disease_labels = st.multiselect("Select Model Prediction Label", data['Model_preds'].unique())

        filtered_data = data[data['Model_preds'].isin(selected_disease_labels)] if selected_disease_labels else data

        st.dataframe(filtered_data, height=200)
        st.divider()

        selected_id = st.selectbox("Select an ID", data[id_column])
        
        selected_index = data[data[id_column] == selected_id].index[0]
        st.write(data[data[id_column] == selected_id])
        
        sliders = []
        col1, col2 = st.columns(2)
        with col1:
            for feature in feature_columns:

                ing_slider = st.slider(label=feature, min_value=data[feature].min().astype(float), max_value=data[feature].max().astype(float),
                                       value=data.loc[selected_index, feature].astype(float),
                                       step=0.1)
                sliders.append(ing_slider)

        with col2:
            col1, col2 = st.columns(2, gap="medium")

            prediction = model.predict([sliders])

            with col1:
                st.markdown("### Model Prediction: **{}**".format(prediction[0]))

            probs = model.predict_proba([sliders])
            probability = probs[0][list(model.classes_).index(prediction[0])]

            with col2:
                st.metric(label="Model Confidence", value="{:.2f} %".format(probability * 100), delta="{:.2f} %".format((probability - 0.5) * 100))

            explainer = lime_tabular.LimeTabularExplainer(np.array(X_train), mode="classification",
                                                          class_names=list(data[target_column].unique())[::-1],
                                                          feature_names=feature_columns)
            explanation = explainer.explain_instance(np.array(sliders), model.predict_proba,
                                                     num_features=len(feature_columns), top_labels=3)

            interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
            st.pyplot(interpretation_fig, use_container_width=True)

            with st.form("Subject Matter Experts Opinion"):
                opinion = st.selectbox("Your Opinion", data[target_column].unique())
                remark = st.text_area("Your Remark")

                submit_button = st.form_submit_button("Submit")

            # Check if opinion.csv file exists, create if not
            csv_filename = 'opinion.csv'
            if not os.path.exists(csv_filename):
                opinion_data = pd.DataFrame(columns=['ID', 'Timestamp', 'SME label', 'SME Remark'])
                opinion_data.to_csv(csv_filename, index=False)

            # Update opinion_data and save to CSV if the submit button is clicked
            if submit_button:
                opinion_data = pd.read_csv('opinion.csv')
                new_row = {
                    'ID': selected_index,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Adding timestamp
                    'SME label': opinion,
                    'SME Remark': remark
                }
                opinion_data = opinion_data.append(new_row, ignore_index=True)
                opinion_data.to_csv(csv_filename, index=False)
                st.toast('Updated!!')
                st.experimental_rerun()

