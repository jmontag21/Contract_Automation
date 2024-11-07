import streamlit as st
import pandas as pd
import os
import numpy as np
from io import BytesIO
import zipfile

# ---------------------------
# Streamlit App Setup
# ---------------------------

st.title("Product Contract Processor")

# Create tabs for the two steps
tab1, tab2 = st.tabs(["Step 1: Generate Combined File", "Step 2: Generate Contract Files"])

# ---------------------------
# STEP 1
# ---------------------------
with tab1:
    st.header("Step 1: Generate Combined Prices and Counts")
    st.write("""
    Upload all required CSV files to generate the COMBINED_PRICES_COUNTS file.
    Required files:
    - HOTEL.csv
    - GYM.csv
    - BUSINESS.csv
    - BUILDING.csv
    - RETRO.csv
    - CRUNCH.csv
    - LIB-HOTEL.csv
    - CUSTOMER TYPE LIST.csv
    - ORDER_DETAIL.csv
    """)

    # File uploader for Step 1
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True, key="step1_files")

    # This dictionary will hold the uploaded data
    uploaded_data = {}

    # Load files as they are uploaded
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file, low_memory=False)
                uploaded_data[uploaded_file.name] = df
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
        st.success("Files uploaded successfully!")

    # Button to run Step 1
    if st.button("Generate Combined File", key="step1_button"):
        required_files = [
            'HOTEL.csv', 'GYM.csv', 'BUSINESS.csv', 'BUILDING.csv',
            'RETRO.csv', 'CRUNCH.csv', 'LIB-HOTEL.csv',
            'CUSTOMER TYPE LIST.csv', 'ORDER_DETAIL.csv'
        ]

        missing_files = [file for file in required_files if file not in uploaded_data]
        
        if missing_files:
            st.error(f"Missing files: {', '.join(missing_files)}")
        else:
            try:
                # ---------------------------
                # Load Data from Uploaded Files
                # ---------------------------
                hotel_df = uploaded_data['HOTEL.csv']
                gym_df = uploaded_data['GYM.csv']
                business_df = uploaded_data['BUSINESS.csv']
                building_df = uploaded_data['BUILDING.csv']
                retro_df = uploaded_data['RETRO.csv']
                crunch_df = uploaded_data['CRUNCH.csv']
                library_df = uploaded_data['LIB-HOTEL.csv']
                customer_type_df = uploaded_data['CUSTOMER TYPE LIST.csv']
                order_detail_df = uploaded_data['ORDER_DETAIL.csv']

                # Remove discontinued items
                hotel_df = hotel_df[hotel_df['Discontinued'].fillna('N') != 'Y']
                gym_df = gym_df[gym_df['Discontinued'].fillna('N') != 'Y']
                business_df = business_df[business_df['Discontinued'].fillna('N') != 'Y']
                building_df = building_df[building_df['Discontinued'].fillna('N') != 'Y']
                retro_df = retro_df[retro_df['Discontinued'].fillna('N') != 'Y']
                crunch_df = crunch_df[crunch_df['Discontinued'].fillna('N') != 'Y']
                library_df = library_df[library_df['Discontinued'].fillna('N') != 'Y']

                # Get list of active ProductIDs
                active_products = set()
                for df in [hotel_df, gym_df, business_df, building_df, retro_df, crunch_df, library_df]:
                    active_products.update(df['ProductID'].dropna().astype(str).unique())

                # Filter order_detail_df to only include active products
                order_detail_df = order_detail_df[order_detail_df['ProductID'].astype(str).isin(active_products)]

                # Basic preprocessing
                dfs = [hotel_df, gym_df, business_df, building_df, retro_df, crunch_df, library_df, order_detail_df]
                for i in range(len(dfs)):
                    df = dfs[i]
                    df.dropna(subset=['ProductID'], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    df['ProductID'] = df['ProductID'].astype(str).str.strip()
                    df = df[df['ProductID'].str.lower() != 'nan']
                    df.reset_index(drop=True, inplace=True)
                    dfs[i] = df

                hotel_df, gym_df, business_df, building_df, retro_df, crunch_df, library_df, order_detail_df = dfs

                # Process customer data
                order_detail_df['CustomerID'] = order_detail_df['CustomerID'].astype(str).str.strip()
                customer_type_df.columns = customer_type_df.columns.str.strip()
                customer_type_df['Customer-CustomerID'] = customer_type_df['Customer-CustomerID'].astype(str).str.strip()

                # Merge customer type information
                order_detail_df = order_detail_df.merge(
                    customer_type_df[['Customer-CustomerID', 'Customer-CustomerType']],
                    left_on='CustomerID',
                    right_on='Customer-CustomerID',
                    how='left'
                )

                order_detail_df['Customer-CustomerType'] = order_detail_df['Customer-CustomerType'].str.upper().str.strip()
                order_detail_df = order_detail_df[order_detail_df['IsSpecialItem'] == 'N']

                # ---------------------------
                # Create Combined DataFrame
                # ---------------------------
                # Start with highest cost for each ProductID across all contracts
                all_costs = pd.DataFrame()
                contracts = {
                    'GYM': gym_df,
                    'HOTEL': hotel_df,
                    'BUILDING': building_df,
                    'BUSINESS': business_df
                }
                
                for df in contracts.values():
                    costs = df[['ProductID', 'Cost']].copy()
                    if all_costs.empty:
                        all_costs = costs
                    else:
                        all_costs = pd.concat([all_costs, costs])
                
                max_costs = all_costs.groupby('ProductID')['Cost'].max().reset_index()
                
                # Initialize combined_df with ProductID and COST
                combined_df = max_costs.copy()
                combined_df.columns = ['ProductID', 'COST']

                # Get detailed purchase information for each product
                purchase_details = []
                for product_id in combined_df['ProductID'].unique():
                    product_orders = order_detail_df[order_detail_df['ProductID'] == product_id]
                    
                    # Get unique customers and their total purchases
                    customer_purchases = product_orders.groupby('CustomerID').agg({
                        'CustomerName': 'first',
                        'Quantity': 'sum'
                    }).reset_index()
                    
                    # Only include customer name if exactly one customer with exactly one purchase
                    if len(customer_purchases) == 1 and customer_purchases['Quantity'].iloc[0] == 1:
                        unique_customer = customer_purchases['CustomerName'].iloc[0]
                    else:
                        unique_customer = ''
                    
                    purchase_details.append({
                        'ProductID': product_id,
                        'UNIQUE_CUSTOMER': unique_customer
                    })
                
                # Convert to DataFrame and merge with combined_df
                purchase_df = pd.DataFrame(purchase_details)
                combined_df = combined_df.merge(purchase_df, on='ProductID', how='left')

                # Add price columns for each contract type
                for contract_name, df in contracts.items():
                    price_col = f"{contract_name}Price"
                    contract_prices = df[['ProductID', 'ContractPrice']].copy()
                    contract_prices.columns = ['ProductID', price_col]
                    combined_df = combined_df.merge(contract_prices, on='ProductID', how='left')

                # Add count columns for each contract type
                for contract_name in contracts.keys():
                    count_col = f"{contract_name}Count"
                    contract_counts = order_detail_df[
                        order_detail_df['Customer-CustomerType'] == contract_name
                    ].groupby('ProductID')['Quantity'].sum().reset_index(name=count_col)
                    combined_df = combined_df.merge(contract_counts, on='ProductID', how='left')

                # Fill NaN values with 0
                combined_df = combined_df.fillna(0)

                # Add REVIEW column
                def get_review(row):
                    prices = [row['GYMPrice'], row['HOTELPrice'], row['BUILDINGPrice'], row['BUSINESSPrice']]
                    if all(price == 0 for price in prices):
                        return "ALL ZEROS"
                    
                    non_zero_prices = [p for p in prices if p > 0]
                    if non_zero_prices:
                        lowest_price = min(non_zero_prices)
                        if row['COST'] > 0:  # Using single COST column
                            margin = (lowest_price - row['COST']) / lowest_price * 100
                            if margin < 10:
                                return "LOW MARGIN"
                    return ""

                combined_df['REVIEW'] = combined_df.apply(get_review, axis=1)

                # Reorder columns to match screenshot
                column_order = [
                    'ProductID', 'UNIQUE_CUSTOMER', 'COST',
                    'GYMPrice', 'HOTELPrice', 'BUILDINGPrice', 'BUSINESSPrice',
                    'GYMCount', 'HOTELCount', 'BUILDINGCount', 'BUSINESSCount',
                    'REVIEW'
                ]
                combined_df = combined_df[column_order]

                # Save the combined DataFrame
                output = BytesIO()
                combined_df.to_csv(output, index=False)
                
                st.success("Combined file generated successfully!")
                
                # Download button for the combined file
                st.download_button(
                    label="Download COMBINED_PRICES_COUNTS File",
                    data=output.getvalue(),
                    file_name="COMBINED_PRICES_COUNTS.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                raise e

# ---------------------------
# STEP 2
# ---------------------------
with tab2:
    st.header("Step 2: Generate Contract Files")
    st.write("Upload the COMBINED_PRICES_COUNTS file to generate individual contract files.")

    # File uploader for Step 2
    combined_file = st.file_uploader("Upload COMBINED_PRICES_COUNTS.csv", type="csv", key="step2_file")

    if combined_file is not None:
        try:
            # Read the combined file
            combined_df = pd.read_csv(combined_file)
            st.success("File uploaded successfully!")

            # Button to generate contract files
            if st.button("Generate Contract Files", key="step2_button"):
                try:
                    # Define contract templates to generate
                    contracts = {
                        'BUILDINGS': ['ProductID', 'BUILDINGPrice'],
                        'BUSINESS': ['ProductID', 'BUSINESSPrice'],
                        'CRUNCH BASE': ['ProductID', 'GYMPrice'],
                        'GYM': ['ProductID', 'GYMPrice'],
                        'HOTEL': ['ProductID', 'HOTELPrice'],
                        'LIBRARY HOTEL COLLECTION': ['ProductID', 'HOTELPrice'],
                        'RETRO FITNESS': ['ProductID', 'GYMPrice']
                    }

                    # Create a ZIP file containing all contract files
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for contract_name, columns in contracts.items():
                            # Create contract specific DataFrame
                            contract_df = combined_df[columns].copy()
                            # Rename price column to standard name
                            contract_df.columns = ['ProductID', 'ContractPrice']
                            # Remove rows where ContractPrice is 0
                            contract_df = contract_df[contract_df['ContractPrice'] > 0]
                            # Save to CSV string
                            csv_data = contract_df.to_csv(index=False)
                            # Add to ZIP
                            zf.writestr(f"{contract_name.replace(' ', '_')}.csv", csv_data)

                    st.success("Contract files generated successfully!")

                    # Download button for the ZIP file
                    st.download_button(
                        label="Download All Contract Files",
                        data=zip_buffer.getvalue(),
                        file_name="contract_files.zip",
                        mime="application/zip"
                    )

                except Exception as e:
                    st.error(f"An error occurred while generating contract files: {str(e)}")
                    raise e
        except Exception as e:
            st.error(f"Error reading the combined file: {str(e)}")