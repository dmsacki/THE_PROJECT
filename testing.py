import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


# Load the pre-trained model for prediction
model = joblib.load('gb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit app setup
st.title('Loan Eligibility Dashboard')

# Sidebar for navigation
st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Introduction', 'Dashboard', 'Recommendation'])

# ==================== INTRODUCTION PAGE =====================
if page == 'Introduction':
    st.title("Introduction")
    st.header("Problem Statement")
    st.write("""
    This project aims to predict the likelihood of credit default for e-commerce clients using machine learning. Xente, an e-commerce platform in Uganda, offers personal loans via its mobile app. With "Buy Now & Pay Later" services, customers can borrow funds for various transactions.

    Loan default prediction is crucial for Xente's profitability and risk management. Using transaction data, this project builds a model to predict whether a customer will default on their loan, helping optimize Xente's credit decision-making process.

    The dataset includes over 2,600 transactions, with the target variable "IsDefaulted" representing whether a loan has been defaulted. The model's goal is to maximize the AUC score, ensuring better identification of high-risk customers and improving financial stability.

    """)
    # Correcting the file extension and ensuring the image is in the same folder
    st.image("Xente image.jpeg", caption="Xente: E-commerce and Credit Solutions in Uganda")

    
# ==================== DASHBOARD PAGE =====================
elif page == 'Dashboard':
    # st.title('Dashboard')

    # Sub-navigation for the Dashboard (EDA and Prediction)
    dashboard_page = st.sidebar.radio('Select Dashboard Section', ['EDA', 'Prediction'])

    if dashboard_page == 'EDA':
        st.header('Exploratory Data Analysis')
        
        # Load the cleaned data for EDA
        data = pd.read_csv('Reduced_Data3.csv')

        # Convert specific integer columns to categorical
        data['IsFinalPayBack'] = data['IsFinalPayBack'].astype('category')
        data['IsThirdPartyConfirmed'] = data['IsThirdPartyConfirmed'].astype('category')
        data['IsDefaulted'] = data['IsDefaulted'].astype('category')  # Convert IsDefaulted to categorical
        data['SubscriptionId'] = data['SubscriptionId'].astype('category')
        data['ProductCategory'] = data['ProductCategory'].astype('category')
        data['InvestorId'] = data['InvestorId'].astype('category')
        data['ProductId'] = data['ProductId'].astype('category')

        # Drop columns only for EDA
        columns_to_drop = ['CustomerId', 'Amount', 'Value','TransactionId','IssuedDateLoan', 'TransactionStartTime', 'DueDate', 'PaidOnDate', 'Currency', 'LoanId',    
        "TransactionStatus", "BatchId", "CurrencyCode", "CountryCode", "ProviderId", "PayBackId", 'ChannelId']
        data = data.drop(columns=columns_to_drop)

        # # Show data preview
        # st.write("Here is a preview of the data:")
        # st.dataframe(data.head())

        # Header for EDA Controls
        st.header('EDA Controls')

        # Move the multiselect box to the sidebar for EDA
        st.sidebar.header('EDA Controls')
        analysis_type = st.sidebar.selectbox('Choose Analysis Type', ['Univariate', 'Bivariate'])

        # Layout with two columns for plotting
        col1, col2 = st.columns(2)

        # Function to plot in two columns alternately
        def plot_in_columns(fig, interpretation, index):
            """Plot the figures in two alternating columns along with interpretation."""
            if index % 2 == 0:
                col1.pyplot(fig)
                col1.markdown(interpretation)
            else:
                col2.pyplot(fig)
                col2.markdown(interpretation)

        plot_index = 0  # Plot index tracker for layout

        if analysis_type == 'Univariate':
            # Univariate plot selection in sidebar
            plot_type = st.sidebar.radio('Choose Univariate Plot Type', [
                'Count Plots',
                'Pie Charts',
                'Histograms'
            ])
            
            # Custom color palette with softer colors
            custom_palette = sns.color_palette(["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22"])

            # Univariate Plots
            if plot_type == 'Count Plots':
                count_plots = ['ProductCategory Count Plot', 'SubscriptionId Count Plot', 'Target Count Plot']

                if 'ProductCategory Count Plot' in count_plots:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(x='ProductCategory', data=data, palette=custom_palette, ax=ax)
                    ax.set_title('Product Category Distribution')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
                    interpretation = "The Product Category Count Plot shows the distribution of different categories of products involved in the loans. The highest count indicates the most frequent product category."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1

                if 'SubscriptionId Count Plot' in count_plots:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(x='SubscriptionId', data=data, palette=custom_palette, ax=ax)
                    ax.set_title('SubscriptionId Distribution')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
                    interpretation = "This plot shows how often each Subscription ID occurs in the dataset. Subscription IDs represent unique customer subscriptions."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1

                if 'Target Count Plot' in count_plots:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(x='IsDefaulted', data=data, palette=custom_palette, ax=ax)
                    ax.set_title('Target Distribution: IsDefaulted')
                    interpretation = "The Target Count Plot illustrates the balance or imbalance of the target variable (IsDefaulted). The large difference between categories indicates an imbalanced dataset."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1

            elif plot_type == 'Pie Charts':
                pie_charts = ['ProductCategory Pie Chart', 'SubscriptionId Pie Chart', 'Target Pie Chart']

                if 'ProductCategory Pie Chart' in pie_charts:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data['ProductCategory'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette('Set1'))
                    ax.set_ylabel('')
                    ax.set_title('Product Category Distribution')
                    interpretation = "The Product Category Pie Chart shows the percentage breakdown of each product category, giving a clearer view of the proportion of loans per category."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1

                if 'SubscriptionId Pie Chart' in pie_charts:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data['SubscriptionId'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette('Set1'))
                    ax.set_ylabel('')
                    ax.set_title('SubscriptionId Distribution')
                    interpretation = "The SubscriptionId Pie Chart visualizes the proportion of loans per Subscription ID, showing the distribution across various customers."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1
                
                if 'Target Pie Chart' in pie_charts:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data['IsDefaulted'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette('Set1'))
                    ax.set_ylabel('')
                    ax.set_title('IsDefaulted Distribution')
                    interpretation = "This pie chart represents the ratio of defaulted vs non-defaulted loans, providing insight into how unbalanced the target classes are."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1

            elif plot_type == 'Histograms':
                histograms = ['Histogram: AmountLoan']

                if 'Histogram: AmountLoan' in histograms:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data['AmountLoan'], kde=False, bins=30, ax=ax, color='skyblue')
                    ax.set_title('Histogram: AmountLoan')
                    interpretation = "The AmountLoan Histogram shows the distribution of loan amounts in the dataset, providing insights into the range and concentration of loan sizes."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1

        elif analysis_type == 'Bivariate':
            # Bivariate plot selection in sidebar
            plot_type = st.sidebar.radio('Choose Bivariate Plot Type', [
                'Categorical vs IsDefaulted',
                'Numerical vs IsDefaulted'
            ])

            custom_palette = sns.color_palette(["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22"])

            if plot_type == 'Categorical vs IsDefaulted':
                categorical_columns = [col for col in data.columns if data[col].dtype == 'category']

                for col in categorical_columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(x=col, hue='IsDefaulted', data=data, palette=custom_palette, ax=ax)
                    ax.set_title(f'{col} vs IsDefaulted')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') 
                    ax.legend(title='Defaulted', loc='upper right')
                    interpretation = f"This plot shows the relationship between {col} and the target variable IsDefaulted, providing insight into how different categories affect loan default rates."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1

            elif plot_type == 'Numerical vs IsDefaulted':
                numerical_columns = [col for col in data.columns if data[col].dtype != 'category' and col != 'IsDefaulted']

                for col in numerical_columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.violinplot(x='IsDefaulted', y=col, data=data, palette=custom_palette, ax=ax)
                    ax.set_title(f'{col} vs IsDefaulted')
                    interpretation = f"This violin plot displays the distribution of {col} against the IsDefaulted variable, providing insights into the variance of this numerical feature for defaulted and non-defaulted loans."
                    plot_in_columns(fig, interpretation, plot_index)
                    plot_index += 1
    elif dashboard_page == 'Prediction':
        st.header('Loan Default Prediction')
        # Prediction code continues here...
        import streamlit as st
        import pandas as pd
        from datetime import datetime
        # Load the cleaned data for EDA and Prediction
        data = pd.read_csv('Reduced_Data.csv')
    # Assuming 'model' is your trained model loaded earlier
    
        # st.title("Loan Default Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
        
            feature_6 = st.number_input("Enter Transaction Value", min_value=0.0)
            feature_7 = st.number_input("Enter Loan Amount", min_value=0.0)

        with col2:
            feature_2 = st.selectbox("Is this last payback installment?", ["Yes", "No"])
            feature_4 = st.selectbox("Specify Product Category", ["airtime", 'retail', 'utility_bill', 'movies', 'financial_services', 'tv', 'data_bundles'])

        with col3:
            feature_5 = st.selectbox("Enter your SubscriptionID", [1, 4, 5, 6, 7])
            feature_3 = st.selectbox("Is Third Party Confirmed?", ["Yes", "No"])

        with col1:
            # Date inputs
            due_date = st.date_input(" Enter Due Date", value=datetime.today())
            
        with col2:
            # Date inputs
            paid_on_date = st.date_input("Enter Paid On Date", value=datetime.today())

        # Calculate GracePeriod
        grace_period = (paid_on_date - due_date).days
        grace_period = max(grace_period, 0)  # Ensure GracePeriod is non-negative

        # Ensure all fields are filled
        if st.button('Make Prediction'):
            if feature_6 is not None and feature_7 is not None:
                # Map input fields to the correct features
                input_data = {
                
                    "Value": feature_6,
                    "AmountLoan": feature_7,
                    "GracePeriod": grace_period,
                    "IsFinalPayBack": 1 if feature_2 == "Yes" else 0,
                    "IsThirdPartyConfirmed": 1 if feature_3 == "Yes" else 0,
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

                # Define the feature columns expected by the model
                
                all_features = ['Value', 'AmountLoan', 'IsFinalPayBack', 'IsThirdPartyConfirmed', 
                    'ProductCategory_airtime', 'ProductCategory_data_bundles', 
                    'ProductCategory_financial_services', 'ProductCategory_movies', 
                    'ProductCategory_retail', 'ProductCategory_tv', 
                    'ProductCategory_utility_bill', 'SubscriptionId_SubscriptionId_1', 
                    'SubscriptionId_SubscriptionId_4', 'SubscriptionId_SubscriptionId_5', 
                    'SubscriptionId_SubscriptionId_6', 'SubscriptionId_SubscriptionId_7', 
                    'GracePeriod']

                # Create DataFrame for prediction input
                input_df = pd.DataFrame([input_data], columns=all_features).fillna(0)

                # Make prediction
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]

            # Display the prediction result and probabilities
            st.subheader(f"Prediction: {'Defaulted' if prediction == 1 else 'Not Defaulted'}")
            st.write(f"Probability of Defaulting: {probabilities[1]:.2f}")
            st.write(f"Probability of Not Defaulting: {probabilities[0]:.2f}")
        else:
            st.error("Please fill in all the input fields.")
            #     # Display the prediction result
            #     st.subheader(f"Prediction: {'Defaulted' if prediction == 1 else 'Not Defaulted'}")
            # else:
            #     st.error("Please fill in all the input fields.")

    # If nothing is selected in the multiselect, display a message with larger font
    # if page == 'EDA' and not selected_graphs:
# ==================== RECOMMENDATION PAGE =====================
elif page == 'Recommendation':
    st.title("Loan Eligibility Dashboard - Recommendations")
    st.write("""
    Based on the loan default prediction model, we suggest the following:

    1. **Enhanced Credit Risk Assessment**: Use the model to identify high-risk clients and minimize default rates, improving financial sustainability.
    
    2. **Model Updates**: Regularly retrain the model with new data to adapt to changing customer behavior and market conditions.
    
    3. **Target High-Risk Categories**: Focus on product categories with higher default rates by adjusting loan amounts or payment terms.
    
    4. **Personalized Loan Offers**: Offer better loan terms to low-risk customers based on their transaction history.
    
    5. **Monitor Subscriptions**: Track repayment patterns linked to subscription IDs to reduce defaults in high-risk groups.
    
    6. **Feedback Loop**: Feed new default data back into the model to improve accuracy over time.

    ### Future Work:
    - Analyze macroeconomic factors and customer demographics to enhance predictions.
    - Explore alternative models or ensemble techniques for better performance.
    """)
