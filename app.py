# Command: Build the Streamlit application for Human Activity Recognition

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)


# ============================================================
# 2. GET THE FOLDER WHERE app.py IS LOCATED
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 3. MODEL FILE PATHS
# ============================================================

# Your model files are inside the rnn.ipynb folder
MODEL_PATH = os.path.join(
    BASE_DIR,
    "rnn.ipynb",
    "rnn_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "rnn.ipynb",
    "scaler.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "rnn.ipynb",
    "label_encoder.pkl"
)


# ============================================================
# 4. CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ rnn_model.keras was not found.\n\n"
        f"Expected location:\n{MODEL_PATH}"
    )

    st.stop()


if not os.path.exists(SCALER_PATH):

    st.error(
        "❌ scaler.pkl was not found.\n\n"
        f"Expected location:\n{SCALER_PATH}"
    )

    st.stop()


if not os.path.exists(ENCODER_PATH):

    st.error(
        "❌ label_encoder.pkl was not found.\n\n"
        f"Expected location:\n{ENCODER_PATH}"
    )

    st.stop()


# ============================================================
# 5. LOAD RNN MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ============================================================
# 6. LOAD SCALER AND LABEL ENCODER
# ============================================================

@st.cache_resource
def load_preprocessing():

    with open(
        SCALER_PATH,
        "rb"
    ) as file:

        scaler = pickle.load(file)


    with open(
        ENCODER_PATH,
        "rb"
    ) as file:

        label_encoder = pickle.load(file)


    return scaler, label_encoder


scaler, label_encoder = load_preprocessing()


# ============================================================
# 7. TITLE
# ============================================================

st.title("🏃 Human Activity Recognition")

st.subheader(
    "RNN-Based Human Activity Prediction"
)

st.write(
    "Upload a CSV file containing the 561 sensor features "
    "to predict human activities using the trained RNN model."
)


st.divider()


# ============================================================
# 8. SUPPORTED ACTIVITIES
# ============================================================

st.subheader("Supported Activities")

activities = label_encoder.classes_

columns = st.columns(3)

for i, activity in enumerate(activities):

    with columns[i % 3]:

        st.info(
            f"**{activity}**"
        )


st.divider()


# ============================================================
# 9. UPLOAD CSV FILE
# ============================================================

st.subheader("📂 Upload CSV Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ============================================================
# 10. PROCESS THE UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        uploaded_file
    )


    st.success(
        "✅ CSV file uploaded successfully!"
    )


    # --------------------------------------------------------
    # Display uploaded dataset
    # --------------------------------------------------------

    st.subheader(
        "Uploaded Dataset"
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )


    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Number of Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Number of Columns",
            df.shape[1]
        )


    # --------------------------------------------------------
    # Remove non-feature columns
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            "Activity",
            "subject"
        ],
        errors="ignore"
    )


    # --------------------------------------------------------
    # Check number of features
    # --------------------------------------------------------

    if X.shape[1] != 561:

        st.error(
            f"❌ Incorrect number of features.\n\n"
            f"Expected: 561 features\n"
            f"Received: {X.shape[1]} features"
        )

        st.stop()


    # --------------------------------------------------------
    # Convert all features to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    if X.isnull().any().any():

        st.error(
            "❌ The uploaded dataset contains "
            "missing or non-numeric values."
        )

        st.stop()


    st.success(
        "✅ Dataset contains exactly 561 valid features."
    )


    # ========================================================
    # 11. PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Activity",
        type="primary"
    ):

        with st.spinner(
            "RNN model is predicting..."
        ):

            # ------------------------------------------------
            # Scale the features
            # ------------------------------------------------

            X_scaled = scaler.transform(
                X
            )


            # ------------------------------------------------
            # Reshape for RNN
            # ------------------------------------------------

            X_rnn = X_scaled.reshape(
                X_scaled.shape[0],
                561,
                1
            )


            # ------------------------------------------------
            # Make predictions
            # ------------------------------------------------

            predictions = model.predict(
                X_rnn,
                verbose=0
            )


            # ------------------------------------------------
            # Get predicted class
            # ------------------------------------------------

            predicted_classes = np.argmax(
                predictions,
                axis=1
            )


            # ------------------------------------------------
            # Convert class numbers to activity names
            # ------------------------------------------------

            predicted_activities = (
                label_encoder.inverse_transform(
                    predicted_classes
                )
            )


            # ------------------------------------------------
            # Calculate confidence
            # ------------------------------------------------

            confidence = (
                np.max(
                    predictions,
                    axis=1
                ) * 100
            )


        # ====================================================
        # 12. CREATE RESULT DATAFRAME
        # ====================================================

        result_df = pd.DataFrame({

            "Predicted Activity":
                predicted_activities,

            "Confidence (%)":
                np.round(
                    confidence,
                    2
                )
        })


        # ====================================================
        # 13. DISPLAY PREDICTION RESULTS
        # ====================================================

        st.success(
            "🎉 Prediction completed successfully!"
        )


        st.subheader(
            "📊 Prediction Results"
        )


        st.dataframe(
            result_df,
            use_container_width=True
        )


        # ====================================================
        # 14. ACTIVITY DISTRIBUTION
        # ====================================================

        st.subheader(
            "📈 Activity Distribution"
        )


        activity_counts = (
            pd.Series(
                predicted_activities
            )
            .value_counts()
        )


        st.bar_chart(
            activity_counts
        )


        # ====================================================
        # 15. MOST COMMON ACTIVITY
        # ====================================================

        most_common_activity = (
            activity_counts.index[0]
        )

        most_common_count = (
            activity_counts.iloc[0]
        )


        st.success(
            f"🏆 Most detected activity: "
            f"**{most_common_activity}** "
            f"({most_common_count} samples)"
        )


        # ====================================================
        # 16. DOWNLOAD RESULTS
        # ====================================================

        result_csv = result_df.to_csv(
            index=False
        )


        st.download_button(
            label="⬇️ Download Predictions",
            data=result_csv,
            file_name="activity_predictions.csv",
            mime="text/csv"
        )


# ============================================================
# 17. NO FILE MESSAGE
# ============================================================

else:

    st.info(
        "👆 Please upload a CSV file to start prediction."
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "Human Activity Recognition | "
    "Recurrent Neural Network (RNN)"
)