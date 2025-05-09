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
        
    # Find the closest price tier - more efficient implementation
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
    - ProductList.csv
    """)

    # File uploader for Step 1
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True, key="step1_files")

    # This dictionary will hold the uploaded data
    uploaded_data = {}

    # Load files as they are uploaded
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Reset the cursor position to the beginning of the file
                uploaded_file.seek(0)
                # Read the CSV file
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
            'CUSTOMER TYPE LIST.csv', 'ORDER_DETAIL.csv', 'ProductList.csv'
        ]

        missing_files = [file for file in required_files if file not in uploaded_data]
        
        if missing_files:
            st.error(f"Missing files: {', '.join(missing_files)}")
        else:
            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                def update_progress(progress_pct, message):
                    progress_bar.progress(progress_pct)
                    status_text.text(message)
                
                # Create a function to safely load and process files
                def safe_load_file(file_name, required_columns=None):
                    try:
                        if file_name not in uploaded_data:
                            st.error(f"Missing file: {file_name}")
                            return None
                            
                        df = uploaded_data[file_name]
                        
                        # Check for required columns
                        if required_columns:
                            missing = [col for col in required_columns if col not in df.columns]
                            if missing:
                                st.warning(f"Missing columns in {file_name}: {', '.join(missing)}")
                                # Add empty columns for missing ones
                                for col in missing:
                                    df[col] = ""
                        
                        return df
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {str(e)}")
                        return None
                
                update_progress(0.05, "Loading product list...")
                
                # Load ProductList first to get discontinued items
                product_list_df = safe_load_file('ProductList.csv', ['ProductID', 'ProductCostDiscontinued'])
                if product_list_df is None:
                    st.error("Failed to load ProductList.csv")
                    raise Exception("Failed to load required file ProductList.csv")
                
                # Convert to category type for more efficient memory usage
                product_list_df['ProductID'] = product_list_df['ProductID'].astype(str).str.strip()
                update_progress(0.1, "Identifying discontinued products...")
                
                discontinued_products = set(product_list_df[
                    product_list_df['ProductCostDiscontinued'] == True
                ]['ProductID'].astype(str).tolist())
                
                update_progress(0.15, "Loading contract files...")
                
                # Load all other files
                hotel_df = safe_load_file('HOTEL.csv')
                gym_df = safe_load_file('GYM.csv')
                business_df = safe_load_file('BUSINESS.csv')
                building_df = safe_load_file('BUILDING.csv')
                retro_df = safe_load_file('RETRO.csv')
                crunch_df = safe_load_file('CRUNCH.csv')
                library_df = safe_load_file('LIB-HOTEL.csv')
                customer_type_df = safe_load_file('CUSTOMER TYPE LIST.csv')
                
                # Load order_detail_df with optimization for large files
                st.write("Loading ORDER_DETAIL.csv (this may take a moment)...")
                
                # Use a temporary file approach to avoid issues with file-like objects
                try:
                    # Check if ORDER_DETAIL.csv exists in uploaded_data
                    if 'ORDER_DETAIL.csv' not in uploaded_data:
                        st.error("ORDER_DETAIL.csv not found in uploaded files")
                        order_detail_df = pd.DataFrame(columns=['ProductID', 'CustomerID', 'CustomerName', 'Quantity', 'IsSpecialItem', 'Price'])
                    else:
                        # Use the DataFrame we already loaded properly during the initial upload
                        order_detail_df = uploaded_data['ORDER_DETAIL.csv']
                        st.write(f"Successfully loaded ORDER_DETAIL.csv with {len(order_detail_df)} rows")
                        
                        # Check if Price column exists
                        if 'Price' in order_detail_df.columns:
                            st.write("Found Price column in ORDER_DETAIL.csv")
                        else:
                            # Look for case-insensitive 'price' in column names
                            price_cols = [col for col in order_detail_df.columns if 'price' in col.lower()]
                            if price_cols:
                                price_col = price_cols[0]
                                st.write(f"Using {price_col} column instead of Price")
                                order_detail_df['Price'] = order_detail_df[price_col]
                            else:
                                # Try to calculate price from other columns
                                if 'ExtendedPrice' in order_detail_df.columns and 'Quantity' in order_detail_df.columns:
                                    st.write("Calculating Price from ExtendedPrice/Quantity")
                                    
                                    # Try to convert to numeric
                                    try:
                                        order_detail_df['ExtendedPrice'] = pd.to_numeric(order_detail_df['ExtendedPrice'], errors='coerce')
                                        order_detail_df['Quantity'] = pd.to_numeric(order_detail_df['Quantity'], errors='coerce')
                                        
                                        # Calculate price where quantity > 0
                                        valid_mask = (order_detail_df['Quantity'] > 0)
                                        order_detail_df.loc[valid_mask, 'Price'] = order_detail_df.loc[valid_mask, 'ExtendedPrice'] / order_detail_df.loc[valid_mask, 'Quantity']
                                        
                                        # Fill NaN values with 0
                                        order_detail_df['Price'] = order_detail_df['Price'].fillna(0)
                                    except Exception as calc_err:
                                        st.error(f"Error calculating Price: {str(calc_err)}")
                                        order_detail_df['Price'] = 0
                                else:
                                    st.warning("Price column not found in ORDER_DETAIL.csv. Using 0 as fallback.")
                                    order_detail_df['Price'] = 0
                        
                        # Make sure all required columns exist
                        required_cols = ['ProductID', 'CustomerID', 'CustomerName', 'Quantity', 'IsSpecialItem', 'Price']
                        for col in required_cols:
                            if col not in order_detail_df.columns:
                                order_detail_df[col] = ''
                        
                        # Keep only needed columns if they exist
                        available_cols = [col for col in required_cols if col in order_detail_df.columns]
                        order_detail_df = order_detail_df[available_cols]
                        
                        # Add any missing columns
                        for col in set(required_cols) - set(available_cols):
                            order_detail_df[col] = ""
                
                except Exception as e:
                    st.error(f"Error processing ORDER_DETAIL.csv: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
                    # Create a minimal DataFrame
                    order_detail_df = pd.DataFrame(columns=['ProductID', 'CustomerID', 'CustomerName', 'Quantity', 'IsSpecialItem', 'Price'])
                
                # Continue with memory optimization
                for col in order_detail_df.columns:
                    if order_detail_df[col].dtype == 'object':
                        order_detail_df[col] = order_detail_df[col].astype('string')

                update_progress(0.25, "Processing contract files...")
                
                # Process using optimized operations
                # Define a function to apply common operations to all contract DataFrames
                def preprocess_contract_df(df, name):
                    st.write(f"Processing {name} contract...")
                    # Convert to string and filter out discontinued products efficiently
                    df['ProductID'] = df['ProductID'].astype(str).str.strip()
                    df = df[~df['ProductID'].isin(discontinued_products)]
                    df = df.dropna(subset=['ProductID'])
                    df = df[df['ProductID'].str.lower() != 'nan']
                    return df

                # Apply preprocessing to all contract DataFrames in a single loop
                contract_dfs = {
                    'HOTEL': hotel_df,
                    'GYM': gym_df,
                    'BUSINESS': business_df,
                    'BUILDING': building_df,
                    'RETRO': retro_df,
                    'CRUNCH': crunch_df,
                    'LIBRARY': library_df
                }

                # Process contract DataFrames in parallel when possible
                for name, df in contract_dfs.items():
                    contract_dfs[name] = preprocess_contract_df(df, name)

                # Reassign filtered DataFrames
                hotel_df = contract_dfs['HOTEL']
                gym_df = contract_dfs['GYM']
                business_df = contract_dfs['BUSINESS']
                building_df = contract_dfs['BUILDING']
                retro_df = contract_dfs['RETRO']
                crunch_df = contract_dfs['CRUNCH']
                library_df = contract_dfs['LIBRARY']

                # Filter order_detail_df
                order_detail_df['ProductID'] = order_detail_df['ProductID'].astype(str)
                update_progress(0.4, "Filtering order details...")
                
                # Remove discontinued products
                order_detail_df = order_detail_df[~order_detail_df['ProductID'].isin(discontinued_products)]
                
                # Filter out special items 
                if 'IsSpecialItem' in order_detail_df.columns:
                    order_detail_df = order_detail_df[order_detail_df['IsSpecialItem'] == 'N']
                
                # Get list of unique ProductIDs from order_detail_df (this will be our main product list)
                all_ordered_products = order_detail_df['ProductID'].unique()
                st.write(f"Found {len(all_ordered_products)} unique products in order details (excluding special items)")
                
                # Instead of limiting to active products, we want ALL products from order_detail
                # This is the key change - we're NOT filtering orders to only include products in base contracts
                
                # Basic preprocessing for order_detail_df
                order_detail_df.dropna(subset=['ProductID'], inplace=True)
                order_detail_df.reset_index(drop=True, inplace=True)
                order_detail_df = order_detail_df[order_detail_df['ProductID'].str.lower() != 'nan']
                order_detail_df.reset_index(drop=True, inplace=True)
                
                # Basic preprocessing for contract DataFrames
                dfs = [hotel_df, gym_df, business_df, building_df, retro_df, crunch_df, library_df]
                for i in range(len(dfs)):
                    df = dfs[i]
                    df.dropna(subset=['ProductID'], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    df['ProductID'] = df['ProductID'].astype(str).str.strip()
                    df = df[df['ProductID'].str.lower() != 'nan']
                    df.reset_index(drop=True, inplace=True)
                    dfs[i] = df

                hotel_df, gym_df, business_df, building_df, retro_df, crunch_df, library_df = dfs

                # Process customer data - with chunking for large datasets
                order_detail_df['CustomerID'] = order_detail_df['CustomerID'].astype(str).str.strip()
                customer_type_df.columns = customer_type_df.columns.str.strip()
                customer_type_df['Customer-CustomerID'] = customer_type_df['Customer-CustomerID'].astype(str).str.strip()

                # Merge customer type information with chunks to prevent memory issues
                update_progress(0.6, "Merging customer data...")
                st.write("Merging customer type information...")
                
                # Convert customer type df to dictionary for faster lookups
                customer_type_dict = dict(zip(
                    customer_type_df['Customer-CustomerID'], 
                    customer_type_df['Customer-CustomerType']
                ))
                
                # Apply the mapping directly instead of merge
                order_detail_df['Customer-CustomerType'] = order_detail_df['CustomerID'].map(customer_type_dict)
                
                # Fill missing values and convert to uppercase
                order_detail_df['Customer-CustomerType'] = order_detail_df['Customer-CustomerType'].fillna('')
                order_detail_df['Customer-CustomerType'] = order_detail_df['Customer-CustomerType'].str.upper().str.strip()
                
                # IMPORTANT: Fix the customer type values to match expected values
                # Map any variations to the standard values
                customer_type_mapping = {
                    'BUILDINGS': 'BUILDING',
                    'HOTELS': 'HOTEL',
                    'GYMS': 'GYM',
                    'BUSINESSES': 'BUSINESS'
                }
                
                # Apply the mapping to standardize customer types
                for old_type, new_type in customer_type_mapping.items():
                    order_detail_df.loc[order_detail_df['Customer-CustomerType'] == old_type, 'Customer-CustomerType'] = new_type
                
                # Create sets of product IDs in each contract for faster lookups
                hotel_products = set(hotel_df['ProductID'])
                gym_products = set(gym_df['ProductID'])
                business_products = set(business_df['ProductID'])
                building_products = set(building_df['ProductID'])
                retro_products = set(retro_df['ProductID'])
                crunch_products = set(crunch_df['ProductID'])
                library_products = set(library_df['ProductID'])
                
                # Filter out special items
                order_detail_df = order_detail_df[order_detail_df['IsSpecialItem'] == 'N']

                # Create Combined DataFrame
                contracts = {
                    'GYM': gym_df,
                    'HOTEL': hotel_df,
                    'BUILDING': building_df,
                    'BUSINESS': business_df
                }

                # Get cost information
                update_progress(0.75, "Calculating prices...")
                product_list_df['ProductID'] = product_list_df['ProductID'].astype(str).str.strip()
                cost_data = product_list_df[['ProductID', 'Cost']].copy()
                cost_data = cost_data.dropna(subset=['Cost'])
                
                # Create a DataFrame with all ordered products
                st.write("Creating combined data frame with all ordered products...")
                combined_df = pd.DataFrame({'ProductID': all_ordered_products})
                
                # Merge with cost data
                max_costs = cost_data.groupby('ProductID')['Cost'].max().reset_index()
                combined_df = combined_df.merge(max_costs, on='ProductID', how='left')
                combined_df.rename(columns={'Cost': 'COST'}, inplace=True)
                combined_df['COST'] = combined_df['COST'].fillna(0)
                
                # Check if product exists in any base contract - using sets for faster lookup
                def get_contract_presence(product_id):
                    contracts_present = []
                    
                    # Check each contract individually
                    if product_id in hotel_products:
                        contracts_present.append("HOTEL")
                    if product_id in gym_products:
                        contracts_present.append("GYM")
                    if product_id in business_products:
                        contracts_present.append("BUSINESS")
                    if product_id in building_products:
                        contracts_present.append("BUILDING")
                    if product_id in retro_products:
                        contracts_present.append("RETRO")
                    if product_id in crunch_products:
                        contracts_present.append("CRUNCH")
                    if product_id in library_products:
                        contracts_present.append("LIBRARY")
                    
                    # Return comma-separated list of contracts or blank if none
                    return ", ".join(contracts_present) if contracts_present else ""
                
                # Apply the function to each product
                product_ids = combined_df['ProductID'].tolist()
                combined_df['On_Base_Contract'] = [get_contract_presence(pid) for pid in product_ids]
                
                # Also add a simple Yes/No column for filtering
                combined_df['On_Any_Contract'] = combined_df['On_Base_Contract'].apply(lambda x: "Yes" if x else "No")
                
                # Count and display statistics
                on_contract = (combined_df['On_Any_Contract'] == 'Yes').sum()
                not_on_contract = (combined_df['On_Any_Contract'] == 'No').sum()
                st.write(f"Products on at least one base contract: {on_contract}")
                st.write(f"Products not on any base contract: {not_on_contract}")

                # Get detailed purchase information more efficiently
                update_progress(0.8, "Gathering purchase details...")
                st.write("Analyzing purchase details...")
                
                # Create a more efficient groupby operation
                product_customer_counts = order_detail_df.groupby(['ProductID', 'CustomerID']).agg({
                    'CustomerName': 'first',
                    'Quantity': 'sum'
                }).reset_index()
                
                # Count unique customers per product
                product_customer_stats = product_customer_counts.groupby('ProductID').agg({
                    'CustomerID': 'count',  # Count of unique customers
                    'Quantity': 'sum'      # Total quantity
                }).reset_index()
                
                # Create a dictionary to look up unique customers
                unique_customers = {}
                
                # Process in smaller chunks to prevent memory issues
                for product_id in product_customer_stats['ProductID'].unique():
                    # Only process products with a single customer
                    customers_for_product = product_customer_stats[product_customer_stats['ProductID'] == product_id]
                    if len(customers_for_product) > 0 and customers_for_product['CustomerID'].values[0] == 1:
                        # Get that one customer's info
                        customer_info = product_customer_counts[product_customer_counts['ProductID'] == product_id]
                        if len(customer_info) == 1 and customer_info['Quantity'].values[0] == 1:
                            unique_customers[product_id] = customer_info['CustomerName'].values[0]
                        else:
                            unique_customers[product_id] = ''
                    else:
                        unique_customers[product_id] = ''
                
                # Add the unique customer information to the combined dataframe
                combined_df['UNIQUE_CUSTOMER'] = combined_df['ProductID'].map(unique_customers).fillna('')

                # Add count columns
                update_progress(0.7, "Adding count information...")
                st.write("Calculating quantity counts by customer type...")
                
                # Initialize count columns with zeros
                combined_df['GYMCount'] = 0
                combined_df['HOTELCount'] = 0 
                combined_df['BUILDINGCount'] = 0
                combined_df['BUSINESSCount'] = 0
                
                # Create count dictionaries for each customer type
                gym_counts = {}
                hotel_counts = {}
                building_counts = {}
                business_counts = {}
                
                # Check if we have the necessary columns
                if 'Customer-CustomerType' in order_detail_df.columns and 'Quantity' in order_detail_df.columns:
                    # Process GYM counts
                    gym_orders = order_detail_df[order_detail_df['Customer-CustomerType'] == 'GYM']
                    for _, row in gym_orders.iterrows():
                        product_id = row['ProductID']
                        quantity = row['Quantity'] if not pd.isna(row['Quantity']) else 0
                        try:
                            quantity = float(quantity)
                            gym_counts[product_id] = gym_counts.get(product_id, 0) + quantity
                        except (ValueError, TypeError):
                            pass
                    
                    # Process HOTEL counts
                    hotel_orders = order_detail_df[order_detail_df['Customer-CustomerType'] == 'HOTEL']
                    for _, row in hotel_orders.iterrows():
                        product_id = row['ProductID']
                        quantity = row['Quantity'] if not pd.isna(row['Quantity']) else 0
                        try:
                            quantity = float(quantity)
                            hotel_counts[product_id] = hotel_counts.get(product_id, 0) + quantity
                        except (ValueError, TypeError):
                            pass
                    
                    # Process BUILDING counts
                    building_orders = order_detail_df[order_detail_df['Customer-CustomerType'] == 'BUILDING']
                    for _, row in building_orders.iterrows():
                        product_id = row['ProductID']
                        quantity = row['Quantity'] if not pd.isna(row['Quantity']) else 0
                        try:
                            quantity = float(quantity)
                            building_counts[product_id] = building_counts.get(product_id, 0) + quantity
                        except (ValueError, TypeError):
                            pass
                    
                    # Process BUSINESS counts
                    business_orders = order_detail_df[order_detail_df['Customer-CustomerType'] == 'BUSINESS']
                    for _, row in business_orders.iterrows():
                        product_id = row['ProductID']
                        quantity = row['Quantity'] if not pd.isna(row['Quantity']) else 0
                        try:
                            quantity = float(quantity)
                            business_counts[product_id] = business_counts.get(product_id, 0) + quantity
                        except (ValueError, TypeError):
                            pass
                else:
                    st.warning("Missing necessary columns for count calculation: 'Customer-CustomerType' or 'Quantity'")
                
                # Debug info
                st.write(f"Found order counts for {len(gym_counts)} products from GYM customers")
                st.write(f"Found order counts for {len(hotel_counts)} products from HOTEL customers")
                st.write(f"Found order counts for {len(building_counts)} products from BUILDING customers")
                st.write(f"Found order counts for {len(business_counts)} products from BUSINESS customers")
                
                # Apply counts to the combined DataFrame
                for i, row in combined_df.iterrows():
                    product_id = row['ProductID']
                    
                    # Set GYM count
                    if product_id in gym_counts:
                        combined_df.at[i, 'GYMCount'] = gym_counts[product_id]
                        
                    # Set HOTEL count
                    if product_id in hotel_counts:
                        combined_df.at[i, 'HOTELCount'] = hotel_counts[product_id]
                        
                    # Set BUILDING count
                    if product_id in building_counts:
                        combined_df.at[i, 'BUILDINGCount'] = building_counts[product_id]
                        
                    # Set BUSINESS count
                    if product_id in business_counts:
                        combined_df.at[i, 'BUSINESSCount'] = business_counts[product_id]

                # Add price columns
                update_progress(0.65, "Adding price information...")
                st.write("Retrieving contract prices and historical order prices...")
                
                # Initialize price columns with zeros
                combined_df['GYMPrice'] = 0
                combined_df['HOTELPrice'] = 0
                combined_df['BUILDINGPrice'] = 0
                combined_df['BUSINESSPrice'] = 0
                
                # Map contract DataFrames to price column names
                price_mappings = {
                    'GYM': 'GYMPrice',
                    'HOTEL': 'HOTELPrice', 
                    'BUILDING': 'BUILDINGPrice',
                    'BUSINESS': 'BUSINESSPrice'
                }
                
                # Function to safely extract price from a contract DataFrame
                def get_prices_from_contract(contract_df, price_column):
                    if 'ContractPrice' not in contract_df.columns:
                        st.warning(f"No 'ContractPrice' column in contract. Using zeros.")
                        return {}
                    
                    # Create a dictionary mapping ProductID to ContractPrice
                    price_dict = {}
                    for _, row in contract_df.iterrows():
                        # Skip empty product IDs
                        if not pd.isna(row['ProductID']) and str(row['ProductID']).strip():
                            product_id = str(row['ProductID']).strip()
                            if not pd.isna(row['ContractPrice']):
                                try:
                                    price = float(row['ContractPrice'])
                                    price_dict[product_id] = price
                                except (ValueError, TypeError):
                                    # If price can't be converted to float, use 0
                                    pass
                    
                    return price_dict
                
                # Get prices from each contract file
                gym_prices = get_prices_from_contract(gym_df, 'ContractPrice')
                hotel_prices = get_prices_from_contract(hotel_df, 'ContractPrice')
                building_prices = get_prices_from_contract(building_df, 'ContractPrice')
                business_prices = get_prices_from_contract(business_df, 'ContractPrice')
                
                # Debug info
                st.write(f"Found {len(gym_prices)} prices in GYM contract")
                st.write(f"Found {len(hotel_prices)} prices in HOTEL contract")
                st.write(f"Found {len(building_prices)} prices in BUILDING contract")
                st.write(f"Found {len(business_prices)} prices in BUSINESS contract")
                
                # Get historical prices from order details for items not in contracts
                # Check if order_detail_df has relevant price columns
                price_columns = ['Price', 'UnitPrice', 'ContractPrice']
                price_column = None
                
                for col in price_columns:
                    if col in order_detail_df.columns:
                        price_column = col
                        st.write(f"Using '{price_column}' column from ORDER_DETAIL as price source")
                        break
                
                if not price_column:
                    st.warning("Warning: No price column found in ORDER_DETAIL.csv. Displaying available columns:")
                    st.write(list(order_detail_df.columns))
                    
                    # Try to find any column with 'price' in the name (case insensitive)
                    price_cols = [col for col in order_detail_df.columns if 'price' in col.lower()]
                    if price_cols:
                        price_column = price_cols[0]
                        st.write(f"Found potential price column: {price_column}")
                    else:
                        st.error("No price column available in ORDER_DETAIL.csv. Unable to extract historical prices.")
                
                # Only process if we found a price column
                order_prices_by_customer_type = {
                    'GYM': {},
                    'HOTEL': {},
                    'BUILDING': {},
                    'BUSINESS': {}
                }
                
                # Create dictionaries to track customer types that have ordered each product
                product_customer_types = {}
                
                if price_column and 'Customer-CustomerType' in order_detail_df.columns:
                    # Show sample data
                    st.write("Sample data from ORDER_DETAIL for debugging:")
                    sample_data = order_detail_df.head(5)
                    st.write(sample_data)
                    
                    # Check for values in the Customer-CustomerType column
                    customer_type_counts = order_detail_df['Customer-CustomerType'].value_counts()
                    st.write("Customer type distribution:")
                    st.write(customer_type_counts)
                    
                    # Filter order_detail_df to non-null prices
                    price_filtered_orders = order_detail_df[~order_detail_df[price_column].isna()]
                    st.write(f"Records with non-null prices: {len(price_filtered_orders)}")
                    
                    # Check price values
                    try:
                        # Convert prices to numeric to see how many valid prices we have
                        numeric_prices = pd.to_numeric(price_filtered_orders[price_column], errors='coerce')
                        valid_prices = numeric_prices[~numeric_prices.isna()]
                        st.write(f"Records with valid numeric prices: {len(valid_prices)}")
                        
                        if len(valid_prices) > 0:
                            st.write(f"Price range: Min={valid_prices.min()}, Max={valid_prices.max()}")
                    except Exception as e:
                        st.error(f"Error analyzing prices: {str(e)}")
                    
                    # First, collect all prices by product ID and customer type
                    product_prices_by_type = {}
                    
                    # Group by ProductID to get prices
                    for product_id, group in price_filtered_orders.groupby('ProductID'):
                        product_prices_by_type[product_id] = {}
                        
                        # Keep track of which customer types ordered this product
                        customer_types_for_product = set()
                        
                        # Process each customer type for this product
                        for customer_type in ['GYM', 'HOTEL', 'BUILDING', 'BUSINESS']:
                            type_group = group[group['Customer-CustomerType'] == customer_type]
                            
                            if len(type_group) > 0:
                                customer_types_for_product.add(customer_type)
                                
                                # Convert prices to numeric, ignoring errors
                                prices = pd.to_numeric(type_group[price_column], errors='coerce')
                                # Filter out NaN values
                                prices = prices[~prices.isna()]
                                
                                if len(prices) > 0:
                                    # Store all prices for this customer type and product
                                    product_prices_by_type[product_id][customer_type] = list(prices)
                        
                        # Store which customer types ordered this product
                        product_customer_types[product_id] = customer_types_for_product
                    
                    # Log how many products we found pricing for
                    st.write(f"Found pricing data for {len(product_prices_by_type)} unique products")
                    
                    # Now apply the pricing rules for each product
                    for product_id, customer_types in product_customer_types.items():
                        price_dict = product_prices_by_type.get(product_id, {})
                        
                        # Rule 1: Building Only
                        if 'BUILDING' in customer_types and len(customer_types) == 1:
                            # If only Building customers bought this
                            if 'BUILDING' in price_dict:
                                # Take the highest price paid by Building customers
                                building_price = max(price_dict['BUILDING'])
                                # Set this price for all contract types
                                order_prices_by_customer_type['BUILDING'][product_id] = building_price
                                order_prices_by_customer_type['HOTEL'][product_id] = building_price
                                order_prices_by_customer_type['BUSINESS'][product_id] = building_price
                                order_prices_by_customer_type['GYM'][product_id] = building_price
                        
                        # Rule 2: Hotel Only
                        elif 'HOTEL' in customer_types and len(customer_types) == 1:
                            # If only Hotel customers bought this
                            if 'HOTEL' in price_dict:
                                # Take the highest price paid by Hotel customers
                                hotel_price = max(price_dict['HOTEL'])
                                # Add to Business, Building, and Gym contracts
                                order_prices_by_customer_type['HOTEL'][product_id] = hotel_price
                                order_prices_by_customer_type['BUSINESS'][product_id] = hotel_price
                                order_prices_by_customer_type['BUILDING'][product_id] = hotel_price
                                order_prices_by_customer_type['GYM'][product_id] = hotel_price
                        
                        # Rule 3: Business Only
                        elif 'BUSINESS' in customer_types and len(customer_types) == 1:
                            # If only Business customers bought this
                            if 'BUSINESS' in price_dict:
                                # Take the highest price paid by Business customers
                                business_price = max(price_dict['BUSINESS'])
                                # Add to Hotel, Building, and Gym contracts
                                order_prices_by_customer_type['BUSINESS'][product_id] = business_price
                                order_prices_by_customer_type['HOTEL'][product_id] = business_price
                                order_prices_by_customer_type['BUILDING'][product_id] = business_price
                                order_prices_by_customer_type['GYM'][product_id] = business_price
                        
                        # Rule 4: Gym Only
                        elif 'GYM' in customer_types and len(customer_types) == 1:
                            # If only Gym customers bought this
                            if 'GYM' in price_dict:
                                # Take the highest price paid by Gym customers
                                gym_price = max(price_dict['GYM'])
                                # Add to Hotel, Building, and Business contracts
                                order_prices_by_customer_type['GYM'][product_id] = gym_price
                                order_prices_by_customer_type['HOTEL'][product_id] = gym_price
                                order_prices_by_customer_type['BUILDING'][product_id] = gym_price
                                order_prices_by_customer_type['BUSINESS'][product_id] = gym_price
                        
                        # Rule 5: Building & Hotel Only
                        elif 'BUILDING' in customer_types and 'HOTEL' in customer_types and len(customer_types) == 2:
                            # If both Building and Hotel customers bought this
                            building_price = max(price_dict.get('BUILDING', [0]))
                            hotel_price = max(price_dict.get('HOTEL', [0]))
                            
                            # Apply the appropriate prices
                            order_prices_by_customer_type['BUILDING'][product_id] = building_price
                            order_prices_by_customer_type['HOTEL'][product_id] = hotel_price
                            # Apply Building price to Business and Gym contracts
                            order_prices_by_customer_type['BUSINESS'][product_id] = hotel_price
                            order_prices_by_customer_type['GYM'][product_id] = hotel_price
                        
                        # Rule 6: Building & Gym
                        elif 'BUILDING' in customer_types and 'GYM' in customer_types and len(customer_types) == 2:
                            # If both Building and Gym customers bought this
                            building_price = max(price_dict.get('BUILDING', [0]))
                            gym_price = max(price_dict.get('GYM', [0]))
                            # Take the higher of the two prices
                            highest_price = max(building_price, gym_price)
                            
                            # Apply Building price to Hotel and Business contracts
                            order_prices_by_customer_type['BUILDING'][product_id] = building_price
                            order_prices_by_customer_type['GYM'][product_id] = gym_price
                            order_prices_by_customer_type['HOTEL'][product_id] = building_price
                            order_prices_by_customer_type['BUSINESS'][product_id] = building_price
                        
                        # Rule 7: Building & Business
                        elif 'BUILDING' in customer_types and 'BUSINESS' in customer_types and len(customer_types) == 2:
                            # If both Building and Business customers bought this
                            building_price = max(price_dict.get('BUILDING', [0]))
                            business_price = max(price_dict.get('BUSINESS', [0]))
                            
                            # Apply the appropriate prices
                            order_prices_by_customer_type['BUILDING'][product_id] = building_price
                            order_prices_by_customer_type['BUSINESS'][product_id] = business_price
                            # Apply Building price to Hotel and Gym
                            order_prices_by_customer_type['HOTEL'][product_id] = building_price
                            order_prices_by_customer_type['GYM'][product_id] = building_price
                        
                        # Additional combinations and default cases
                        else:
                            # For any other combination, use the highest price per customer type
                            for ctype, prices in price_dict.items():
                                if prices:
                                    order_prices_by_customer_type[ctype][product_id] = max(prices)
                    
                    # Print statistics
                    for customer_type, prices in order_prices_by_customer_type.items():
                        st.write(f"Found {len(prices)} historical prices for {customer_type} customers")
                
                # Apply prices to combined DataFrame
                for i, row in combined_df.iterrows():
                    product_id = row['ProductID']
                    
                    # Get On_Any_Contract value - ensure it exists and handle safely
                    is_on_contract = row['On_Any_Contract'] == 'Yes'
                    
                    # Set GYM price
                    if product_id in gym_prices:
                        # Always prefer contract prices when available
                        combined_df.at[i, 'GYMPrice'] = gym_prices[product_id]
                    elif not is_on_contract and product_id in order_prices_by_customer_type['GYM']:
                        # For items not on contract, use historical prices 
                        combined_df.at[i, 'GYMPrice'] = order_prices_by_customer_type['GYM'][product_id]
                        
                    # Set HOTEL price
                    if product_id in hotel_prices:
                        combined_df.at[i, 'HOTELPrice'] = hotel_prices[product_id]
                    elif not is_on_contract and product_id in order_prices_by_customer_type['HOTEL']:
                        combined_df.at[i, 'HOTELPrice'] = order_prices_by_customer_type['HOTEL'][product_id]
                        
                    # Set BUILDING price
                    if product_id in building_prices:
                        combined_df.at[i, 'BUILDINGPrice'] = building_prices[product_id]
                    elif not is_on_contract and product_id in order_prices_by_customer_type['BUILDING']:
                        combined_df.at[i, 'BUILDINGPrice'] = order_prices_by_customer_type['BUILDING'][product_id]
                        
                    # Set BUSINESS price
                    if product_id in business_prices:
                        combined_df.at[i, 'BUSINESSPrice'] = business_prices[product_id]
                    elif not is_on_contract and product_id in order_prices_by_customer_type['BUSINESS']:
                        combined_df.at[i, 'BUSINESSPrice'] = order_prices_by_customer_type['BUSINESS'][product_id]

                # Add price information directly from ORDER_DETAIL for items not on contracts
                # This is our final fallback approach for any items still without prices
                update_progress(0.85, "Final price check...")
                st.write("Final check for products without prices...")
                
                # Now that we've definitely created the price columns, we can check for zero prices
                # Identify products without prices (all four price columns are 0)
                zero_price_mask = (
                    (combined_df['GYMPrice'] == 0) &
                    (combined_df['HOTELPrice'] == 0) &
                    (combined_df['BUILDINGPrice'] == 0) &
                    (combined_df['BUSINESSPrice'] == 0)
                )
                
                # Filter to get only products not on contracts with zero prices
                zero_price_products = combined_df[
                    (combined_df['On_Any_Contract'] == 'No') &
                    zero_price_mask
                ]
                
                st.write(f"Found {len(zero_price_products)} products not on contracts with zero prices")
                
                if len(zero_price_products) > 0 and 'Price' in order_detail_df.columns:
                    # Get the list of product IDs
                    zero_price_product_ids = zero_price_products['ProductID'].tolist()
                    
                    # Extract all order details for these products
                    direct_price_orders = order_detail_df[
                        (order_detail_df['ProductID'].isin(zero_price_product_ids)) &
                        (~order_detail_df['Price'].isna())
                    ]
                    
                    st.write(f"Found {len(direct_price_orders)} order records with prices for these products")
                    
                    # Create a direct pricing dictionary
                    product_max_prices = {}
                    
                    # Process each product to get maximum price
                    for product_id, group in direct_price_orders.groupby('ProductID'):
                        try:
                            prices = pd.to_numeric(group['Price'], errors='coerce')
                            valid_prices = prices[~prices.isna()]
                            
                            if len(valid_prices) > 0:
                                max_price = valid_prices.max()
                                product_max_prices[product_id] = max_price
                        except Exception as e:
                            st.write(f"Error processing price for {product_id}: {str(e)}")
                    
                    # Apply these prices to all customer types
                    updated_count = 0
                    for i, row in combined_df.iterrows():
                        if row['ProductID'] in product_max_prices and all(row[col] == 0 for col in ['GYMPrice', 'HOTELPrice', 'BUILDINGPrice', 'BUSINESSPrice']):
                            price = product_max_prices[row['ProductID']]
                            
                            # Apply the same price to all customer types
                            combined_df.at[i, 'GYMPrice'] = price
                            combined_df.at[i, 'HOTELPrice'] = price
                            combined_df.at[i, 'BUILDINGPrice'] = price
                            combined_df.at[i, 'BUSINESSPrice'] = price
                            updated_count += 1
                    
                    # Updated count products with direct price extraction
                    st.write(f"Updated {updated_count} products with direct price extraction")

                # Add REVIEW column
                update_progress(0.95, "Adding review flags...")
                st.write("Adding review flags...")
                
                def get_review(row):
                    prices = [row['GYMPrice'], row['HOTELPrice'], row['BUILDINGPrice'], row['BUSINESSPrice']]
                    if all(price == 0 for price in prices):
                        return "ALL ZEROS"
                    
                    non_zero_prices = [p for p in prices if p > 0]
                    if non_zero_prices:
                        lowest_price = min(non_zero_prices)
                        if row['COST'] > 0:
                            margin = (lowest_price - row['COST']) / lowest_price * 100
                            if margin < 10:
                                return "LOW MARGIN"
                    return ""

                # Vectorized approach for review calculation
                def compute_reviews(df):
                    reviews = []
                    for _, row in df.iterrows():
                        prices = [row['GYMPrice'], row['HOTELPrice'], row['BUILDINGPrice'], row['BUSINESSPrice']]
                        if all(price == 0 for price in prices):
                            reviews.append("ALL ZEROS")
                            continue
                            
                        non_zero_prices = [p for p in prices if p > 0]
                        if non_zero_prices and row['COST'] > 0:
                            lowest_price = min(non_zero_prices)
                            margin = (lowest_price - row['COST']) / lowest_price * 100
                            if margin < 10:
                                reviews.append("LOW MARGIN")
                                continue
                        reviews.append("")
                    return reviews
                
                combined_df['REVIEW'] = compute_reviews(combined_df)

                # Reorder columns
                column_order = [
                    'ProductID', 'UNIQUE_CUSTOMER', 'COST',
                    'GYMPrice', 'HOTELPrice', 'BUILDINGPrice', 'BUSINESSPrice',
                    'GYMCount', 'HOTELCount', 'BUILDINGCount', 'BUSINESSCount',
                    'On_Base_Contract', 'On_Any_Contract', 'REVIEW'
                ]
                combined_df = combined_df[column_order]

                update_progress(0.9, "Creating output file...")
                
                # Debugging - check the content of the combined dataframe
                st.write("Checking data quality...")
                # Check items marked as 'No' in On_Base_Contract
                no_contract_items = combined_df[combined_df['On_Base_Contract'] == 'No']
                st.write(f"Items not on base contract: {len(no_contract_items)}")
                
                # Check if these items have prices in any contract
                has_price = no_contract_items[
                    (no_contract_items['GYMPrice'] > 0) |
                    (no_contract_items['HOTELPrice'] > 0) |
                    (no_contract_items['BUILDINGPrice'] > 0) |
                    (no_contract_items['BUSINESSPrice'] > 0)
                ]
                st.write(f"Items not on contract but having prices: {len(has_price)}")
                
                # Fix inconsistency in "On_Base_Contract" logic
                def correct_base_contract_status(row):
                    # If any price is greater than 0, it should be on a contract
                    if (row['GYMPrice'] > 0 or row['HOTELPrice'] > 0 or 
                        row['BUILDINGPrice'] > 0 or row['BUSINESSPrice'] > 0):
                        # Check if it was incorrectly marked as not on contract
                        if row['On_Base_Contract'] == 'No':
                            # Display some of these inconsistent items for debugging
                            return 'Yes'
                    return row['On_Base_Contract']
                
                # Log some examples of inconsistent entries
                inconsistent = no_contract_items[
                    (no_contract_items['GYMPrice'] > 0) |
                    (no_contract_items['HOTELPrice'] > 0) |
                    (no_contract_items['BUILDINGPrice'] > 0) |
                    (no_contract_items['BUSINESSPrice'] > 0)
                ]
                if len(inconsistent) > 0:
                    st.write("Examples of items marked as 'No' but having prices:")
                    st.write(inconsistent.head())
                
                # Fix the On_Base_Contract values based on price data
                combined_df['On_Base_Contract'] = combined_df.apply(correct_base_contract_status, axis=1)
                
                # Save the combined DataFrame
                output = BytesIO()
                combined_df.to_csv(output, index=False)
                
                update_progress(1.0, "Complete!")
                
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
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
                raise e

# ---------------------------
# STEP 2
# ---------------------------
with tab2:
    st.header("Step 2: Generate Contract Files")
    st.write("Upload the COMBINED_PRICES_COUNTS file to generate individual contract files.")

    combined_file = st.file_uploader("Upload COMBINED_PRICES_COUNTS.csv", type="csv", key="step2_file")

    if combined_file is not None:
        try:
            combined_df = pd.read_csv(combined_file)
            st.success("File uploaded successfully!")

            if st.button("Generate Contract Files", key="step2_button"):
                try:
                    contracts = {
                        'BUILDINGS': ['ProductID', 'BUILDINGPrice'],
                        'BUSINESS': ['ProductID', 'BUSINESSPrice'],
                        'CRUNCH BASE': ['ProductID', 'GYMPrice'],
                        'GYM': ['ProductID', 'GYMPrice'],
                        'HOTEL': ['ProductID', 'HOTELPrice'],
                        'LIBRARY HOTEL COLLECTION': ['ProductID', 'HOTELPrice'],
                        'RETRO FITNESS': ['ProductID', 'GYMPrice']
                    }

                    # Add a field for analysis
                    if 'On_Base_Contract' not in combined_df.columns:
                        st.error("The combined file is missing the 'On_Base_Contract' column. Please regenerate the combined file.")
                        raise Exception("Missing 'On_Base_Contract' column")
                    
                    # Debug what we're working with
                    st.write(f"Total products in combined file: {len(combined_df)}")
                    
                    # Check if On_Any_Contract column exists
                    if 'On_Any_Contract' not in combined_df.columns:
                        st.error("The combined file is missing the 'On_Any_Contract' column. Please regenerate the combined file.")
                        raise Exception("Missing 'On_Any_Contract' column")
                    
                    not_on_contract = combined_df[combined_df['On_Any_Contract'] == "No"]
                    st.write(f"Products not on any base contract: {len(not_on_contract)}")
                    
                    # Count products with prices for each contract type
                    for contract_name, columns in contracts.items():
                        price_column = columns[1]
                        priced_products = not_on_contract[not_on_contract[price_column] > 0]
                        st.write(f"Products for {contract_name} with prices > 0: {len(priced_products)}")
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for contract_name, columns in contracts.items():
                            # Only work with products NOT already on a base contract
                            filtered_df = combined_df[combined_df['On_Any_Contract'] == "No"]
                            
                            # Extract the required columns
                            contract_df = filtered_df[columns].copy()
                            contract_df.columns = ['ProductID', 'ContractPrice']
                            
                            # Filter out zero prices
                            contract_df = contract_df[contract_df['ContractPrice'] > 0]
                            
                            st.write(f"Adding {len(contract_df)} products to {contract_name}.csv")
                            
                            csv_data = contract_df.to_csv(index=False)
                            zf.writestr(f"{contract_name.replace(' ', '_')}.csv", csv_data)

                    st.success("Contract files generated successfully!")

                    st.download_button(
                        label="Download All Contract Files",
                        data=zip_buffer.getvalue(),
                        file_name="contract_files.zip",
                        mime="application/zip"
                    )

                except Exception as e:
                    st.error(f"An error occurred while generating contract files: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
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
                        ⚠️ The estimated price (${estimated_price:.2f}) exceeds our maximum price tier of ${max(PRICE_TIERS):.2f}.
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