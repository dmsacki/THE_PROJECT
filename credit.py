import streamlit as st
import pandas as pd
import joblib

# Load the data
data = pd.read_csv('Data.csv')

# Load the saved best model
best_model = joblib.load('best_model.pkl')

# Sidebar for navigation
st.sidebar.title("Credit Default Prediction Dashboard")
section = st.sidebar.radio("Select a section:", ["EDA", "Prediction"])

# ============= EDA Section =============
if section == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Graph selection at the top
    st.subheader("Select Graph to Display")
    graph_option = st.selectbox(
        "Choose a graph:",
        ["Target Distribution", "Last Installment Distribution", "SubscriptionId_vs_IsFinalPayBack", "Distribution of SubscriptionId by AmountLoan", "Distribution of InvestorId with IsDefaulted"]
    )

    # Displaying graphs based on selection
    if graph_option == "Target Distribution":
        st.image('Target Distribution.png')
        st.markdown("""
        <div style='font-size:24px; line-height:1.5;'>
        **Distribution of Is Defaulted**<br><br>
        This graph shows the distribution of the target variable,<br>
        which indicates whether a person defaulted or not.<br>
        It helps understand the class imbalance in the dataset.
        </div>
        """, unsafe_allow_html=True)

    elif graph_option == "Last Installment Distribution":
        st.image('IsFinalPayBack.png')
        st.markdown("""
        <div style='font-size:24px; line-height:1.5;'>
        **Distribution of Last Installment**<br><br>
        This graph displays the distribution of 'Is Final Payback'.<br>
        It shows how many times the final payback was successfully made or not.<br>
        This helps to analyze the patterns in final repayment status.
        </div>
        """, unsafe_allow_html=True)

    elif graph_option == "SubscriptionId_vs_IsFinalPayBack":
        st.image('SubscriptionId_vs_IsFinalPayBack.png')
        st.markdown("""
        <div style='font-size:24px; line-height:1.5;'>
        **Distribution of SubscriptionId vs IsFinalPayBack**<br><br>
        This graph shows the proportion of final loan repayments for each Subscription ID.<br>
        It helps to identify patterns in repayment behavior and the likelihood of final paybacks.
        </div>
        """, unsafe_allow_html=True)

    elif graph_option == "Distribution of SubscriptionId by AmountLoan":
        st.image('SubscriptionId_vs_AmountLoan.png')
        st.markdown("""
        <div style='font-size:24px; line-height:1.5;'>
        **Distribution of SubscriptionId by AmountLoan**<br><br>
        This graph shows how average loan amounts vary by Subscription ID.<br>
        It helps identify which subscriptions are linked to higher or lower loan amounts.
        </div>
        """, unsafe_allow_html=True)

    elif graph_option == "Distribution of InvestorId with IsDefaulted":
        st.image('InvestorId_vs_IsDefaulted.png')
        st.markdown("""
        <div style='font-size:24px; line-height:1.5;'>
        This graph illustrates the distribution of default rates across different investors.<br>
        It shows how the likelihood of default varies with each Investor ID.<br>
        This helps to identify which investors are associated with higher or lower default rates.
        </div>
        """, unsafe_allow_html=True)

# ============= PREDICTION PAGE =============
elif section == "Prediction":
    st.title("Loan Default Prediction")

    # Input fields in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        feature_1 = st.number_input("Enter value of transaction", min_value=0.0, step=0.01)
        feature_7 = st.number_input("Enter Loan Amount", min_value=0.0, step=0.01)

    with col2:
        feature_2 = st.selectbox("Is it a last Payback installment? (0 = No, 1 = Yes)", [0, 1])
        feature_4 = st.selectbox("Enter Product Category", ["airtime", 'retail', 'utility_bill', 'movies', 'financial_services', 'tv', 'data_bundles'])

    with col3:
        feature_5 = st.selectbox("Enter SubscriptionID", [1, 4, 5, 6, 7])
        feature_3 = st.selectbox("Is Third Party Confirmed? (0 = No, 1 = Yes)", [0, 1])

    # Define the features that were used during training
    all_features = ['Value', 'AmountLoan',
                    'IsFinalPayBack', 'IsThirdPartyConfirmed',
                    'ProductCategory_airtime', 'ProductCategory_movies', 'ProductCategory_financial_services',
                    'ProductCategory_retail', 'ProductCategory_utility_bill', 'ProductCategory_tv',
                    'ProductCategory_data_bundles',
                    'SubscriptionId_SubscriptionId_1', 'SubscriptionId_SubscriptionId_4', 'SubscriptionId_SubscriptionId_5',
                    'SubscriptionId_SubscriptionId_6', 'SubscriptionId_SubscriptionId_7']

    # Prediction section
    if st.button('Make Prediction'):
        input_data = {
            "Value": feature_1,
            "AmountLoan": feature_7,
            "IsFinalPayBack": feature_2,
            "IsThirdPartyConfirmed": feature_3,
            "ProductCategory_airtime": 1 if feature_4 == "airtime" else 0,
            "ProductCategory_movies": 1 if feature_4 == "movies" else 0,
            "ProductCategory_financial_services": 1 if feature_4 == "financial_services" else 0,
            "ProductCategory_retail": 1 if feature_4 == "retail" else 0,
            "ProductCategory_utility_bill": 1 if feature_4 == "utility_bill" else 0,
            "ProductCategory_tv": 1 if feature_4 == "tv" else 0,
            "ProductCategory_data_bundles": 1 if feature_4 == "data_bundles" else 0,
            'SubscriptionId_SubscriptionId_1': 1 if feature_5 == 1 else 0,
            'SubscriptionId_SubscriptionId_4': 1 if feature_5 == 4 else 0,
            'SubscriptionId_SubscriptionId_5': 1 if feature_5 == 5 else 0,
            'SubscriptionId_SubscriptionId_6': 1 if feature_5 == 6 else 0,
            'SubscriptionId_SubscriptionId_7': 1 if feature_5 == 7 else 0,
        }

        # Create DataFrame for prediction input
        input_df = pd.DataFrame([input_data], columns=all_features).fillna(0)

        # Make prediction (get probability)
        prediction_proba = best_model.predict_proba(input_df)[0][1]  # Probability of default

        # Display the prediction result
        st.subheader(f"Prediction: {'Defaulted' if prediction_proba > 0.5 else 'Not Defaulted'}")
