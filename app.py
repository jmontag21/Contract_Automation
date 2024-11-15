import streamlit as st
import pandas as pd
import os
import numpy as np
from io import BytesIO
import zipfile

# ---------------------------
# Streamlit App Setup
# ---------------------------

# Add these at the top of your file with other imports
PRICE_TIERS = [
    1.25, 2.25, 3.25, 4.95, 5.95, 7.95, 9.99, 12.95, 15.5, 19.99, 21.5, 24.5, 28.5, 29.99,
    31.5, 34.5, 39.99, 45.0, 48.5, 59.99, 68.95, 79.99, 96.5, 99.9, 108.5, 118.5, 124.5,
    135.0, 148.5, 179.99, 225.0, 275.0, 299.0, 325.0, 375.0, 395.0, 425.0, 475.0, 495.0,
    525.0, 575.0, 595.0, 625.0, 675.0, 695.0, 725.0, 775.0, 795.0, 825.0, 875.0, 895.0,
    925.0, 975.0, 995.0, 1025.0, 1075.0, 1095.0, 1125.0, 1175.0, 1195.0, 1225.0, 1275.0,
    1295.0, 1325.0, 1375.0, 1395.0, 1425.0, 1475.0, 1495.0, 1525.0, 1575.0, 1595.0, 1625.0,
    1675.0, 1695.0, 1725.0, 1775.0, 1795.0, 1825.0, 1875.0, 1895.0, 1925.0, 1975.0, 1995.0,
    2025.0, 2075.0, 2095.0, 2125.0, 2175.0, 2195.0, 2225.0, 2275.0, 2295.0, 2325.0, 2375.0,
    2395.0, 2425.0, 2475.0, 2495.0, 2525.0, 2575.0, 2595.0, 2625.0, 2675.0, 2695.0, 2725.0,
    2775.0, 2795.0, 2825.0, 2875.0, 2895.0, 2925.0, 2975.0, 2995.0, 3025.0, 3075.0, 3095.0,
    3125.0, 3175.0, 3195.0, 3225.0, 3275.0, 3295.0, 3325.0, 3375.0, 3395.0, 3425.0, 3475.0,
    3495.0, 3525.0, 3575.0, 3595.0, 3625.0, 3675.0, 3695.0, 3725.0, 3775.0, 3795.0, 3825.0,
    3875.0, 3895.0, 3925.0, 3975.0, 3995.0, 4025.0, 4075.0, 4095.0, 4125.0, 4175.0, 4195.0,
    4225.0, 4275.0, 4295.0, 4325.0, 4375.0, 4395.0, 4425.0, 4475.0, 4495.0, 4525.0, 4575.0,
    4595.0, 4625.0, 4675.0, 4695.0, 4725.0, 4775.0, 4795.0, 4825.0, 4875.0, 4895.0, 4925.0,
    4975.0, 4995.0, 5025.0, 5075.0, 5095.0, 5125.0, 5175.0, 5195.0, 5225.0, 5275.0, 5295.0,
    5325.0, 5375.0, 5395.0, 5425.0, 5475.0, 5495.0, 5525.0, 5575.0, 5595.0, 5625.0, 5675.0,
    5695.0, 5725.0, 5775.0, 5795.0, 5825.0, 5875.0, 5895.0, 5925.0, 5975.0, 5995.0, 6025.0,
    6075.0, 6095.0, 6125.0, 6175.0, 6195.0, 6225.0, 6275.0, 6295.0, 6325.0, 6375.0, 6395.0,
    6425.0, 6475.0, 6495.0, 6525.0, 6575.0, 6595.0, 6625.0, 6675.0, 6695.0, 6725.0, 6775.0,
    6795.0, 6825.0, 6875.0, 6895.0, 6925.0, 6975.0, 6995.0, 7025.0, 7075.0, 7095.0, 7125.0,
    7175.0, 7195.0, 7225.0, 7275.0, 7295.0, 7325.0, 7375.0, 7395.0, 7425.0, 7475.0, 7495.0,
    7525.0, 7575.0, 7595.0, 7625.0, 7675.0, 7695.0, 7725.0, 7775.0, 7795.0, 7825.0, 7875.0,
    7895.0, 7925.0, 7975.0, 7995.0, 8025.0, 8075.0, 8095.0, 8125.0, 8175.0, 8195.0, 8225.0,
    8275.0, 8295.0, 8325.0, 8375.0, 8395.0, 8425.0, 8475.0, 8495.0, 8525.0, 8575.0, 8595.0,
    8625.0, 8675.0, 8695.0, 8725.0, 8775.0, 8795.0, 8825.0, 8875.0, 8895.0, 8925.0, 8975.0,
    8995.0, 9025.0, 9075.0, 9095.0, 9125.0, 9175.0, 9195.0, 9225.0, 9275.0, 9295.0, 9325.0,
    9375.0, 9395.0, 9425.0, 9475.0, 9495.0, 9525.0, 9575.0, 9595.0, 9625.0, 9675.0, 9695.0,
    9725.0, 9775.0, 9795.0, 9825.0, 9875.0, 9895.0, 9925.0, 9975.0, 9995.0, 10025.0, 10075.0,
    10095.0, 10125.0, 10175.0, 10195.0, 10225.0, 10275.0, 10295.0, 10325.0, 10375.0, 10395.0,
    10425.0, 10475.0, 10495.0, 10525.0, 10575.0, 10595.0, 10625.0, 10675.0, 10695.0, 10725.0,
    10775.0, 10795.0, 10825.0, 10875.0, 10895.0, 10925.0, 10975.0, 10995.0, 11025.0, 11075.0,
    11095.0, 11125.0, 11175.0, 11195.0, 11225.0, 11275.0, 11295.0, 11325.0, 11375.0, 11395.0,
    11425.0, 11475.0, 11495.0, 11525.0, 11575.0, 11595.0, 11625.0, 11675.0, 11695.0, 11725.0,
    11775.0, 11795.0, 11825.0, 11875.0, 11895.0, 11925.0, 11975.0, 11995.0, 12025.0, 12075.0,
    12095.0, 12125.0, 12175.0, 12195.0, 12225.0, 12275.0, 12295.0, 12325.0, 12375.0, 12395.0,
    12425.0, 12475.0, 12495.0, 12525.0, 12575.0, 12595.0, 12625.0, 12675.0, 12695.0, 12725.0,
    12775.0, 12795.0, 12825.0, 12875.0, 12895.0, 12925.0, 12975.0, 12995.0, 13025.0, 13075.0,
    13095.0, 13125.0, 13175.0, 13195.0, 13225.0, 13275.0, 13295.0, 13325.0, 13375.0, 13395.0,
    13425.0, 13475.0, 13495.0, 13525.0, 13575.0, 13595.0, 13625.0, 13675.0, 13695.0, 13725.0,
    13775.0, 13795.0, 13825.0, 13875.0, 13895.0, 13925.0, 13975.0, 13995.0, 14025.0, 14075.0,
    14095.0, 14125.0, 14175.0, 14195.0, 14225.0, 14275.0, 14295.0, 14325.0, 14375.0, 14395.0,
    14425.0, 14475.0, 14495.0, 14525.0, 14575.0, 14595.0, 14625.0, 14675.0, 14695.0, 14725.0,
    14775.0, 14795.0, 14825.0, 14875.0, 14895.0, 14925.0, 14975.0, 14995.0
]

def calculate_simons_price(cost):
    """
    Calculate Simon's Price based on cost and predefined price tiers.
    """
    if cost <= 0:
        return 0, 0
        
    # Calculate 40% markup
    estimated_price = cost / 0.60
    
    # Check if estimated price exceeds maximum tier
    max_tier = max(PRICE_TIERS)
    if estimated_price > max_tier:
        return estimated_price, None
        
    # Find the closest price tier
    closest_tier = min(PRICE_TIERS, key=lambda x: abs(x - estimated_price))
    
    return estimated_price, closest_tier

st.title("Product Contract Processor")

# Create tabs for the two steps
# Update your tabs creation to include the third tab
tab1, tab2, tab3 = st.tabs([
    "Step 1: Generate Combined File", 
    "Step 2: Generate Contract Files",
    "Simon's Price Calculator"
])

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


# ---------------------------
# STEP 3: Simon's Price Calculator
# ---------------------------
with tab3:
    st.header("Simon's Price Calculator")
    
    # Input for cost with improved styling
    st.write("Enter Item Cost ($)")
    cost = st.number_input("", min_value=0.0, value=0.0, step=0.01, key="cost_input", label_visibility="collapsed")
    
    # Calculate button with custom styling
    if st.button("Calculate Price", key="calculate_price", type="primary"):
        if cost > 0:
            estimated_price, final_price = calculate_simons_price(cost)
            
            # Create two columns for Cost and Estimated Price
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Cost")
                st.markdown(f"<h2>${cost:.2f}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("Estimated Price (40% Markup)")
                st.markdown(f"<h2>${estimated_price:.2f}</h2>", unsafe_allow_html=True)
            
            if final_price is None:
                # Custom warning message with better formatting
                st.markdown(
                    f"""
                    <div style='padding: 1rem; background-color: rgba(255, 193, 7, 0.2); border-radius: 0.5rem; margin: 1rem 0;'>
                        ⚠️ The estimated price ({estimated_price:.2f}) exceeds our maximum price tier of ${max(PRICE_TIERS):.2f}.
                        Please consult with management for pricing.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown("Final Price (Nearest Tier)")
                st.markdown(f"<h2>${final_price:.2f}</h2>", unsafe_allow_html=True)
                actual_margin = ((final_price - cost) / final_price) * 100
                st.markdown(f"Actual Margin: {actual_margin:.1f}%")
        else:
            st.warning("Please enter a cost greater than $0.")