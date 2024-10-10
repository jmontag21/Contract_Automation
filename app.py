# Import necessary libraries
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

st.write("""
Upload all the required CSV files, and the app will process them to generate the required templates.
Make sure you upload the following CSV files:
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

# Allow users to upload multiple files
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True, key="file_uploader_1")

# This dictionary will hold the uploaded data
uploaded_data = {}

# Load files as they are uploaded
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            uploaded_data[uploaded_file.name] = df
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
    st.success("Files uploaded successfully!")

# Button to run the processing script
if st.button("Run Processing Script", key="run_button_1"):

    # ---------------------------
    # Ensure All Required Files Are Uploaded
    # ---------------------------
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

            # ---------------------------
            # Data Preprocessing
            # ---------------------------
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

            order_detail_df['CustomerID'] = order_detail_df['CustomerID'].astype(str).str.strip()
            customer_type_df.columns = customer_type_df.columns.str.strip()
            customer_type_df['Customer-CustomerID'] = customer_type_df['Customer-CustomerID'].astype(str).str.strip()

            order_detail_df = order_detail_df.merge(
                customer_type_df[['Customer-CustomerID', 'Customer-CustomerType']],
                left_on='CustomerID',
                right_on='Customer-CustomerID',
                how='left'
            )

            order_detail_df['Customer-CustomerType'] = order_detail_df['Customer-CustomerType'].str.upper().str.strip()
            order_detail_df = order_detail_df[order_detail_df['IsSpecialItem'] == 'N']

            # ---------------------------
            # Identify New Products
            # ---------------------------
            base_contracts = {
                'BUILDINGS': building_df,
                'HOTEL': hotel_df,
                'GYM': gym_df,
                'BUSINESS': business_df
            }

            base_contract_products = set()
            for df in base_contracts.values():
                base_contract_products.update(df['ProductID'].unique())

            purchased_products = set(order_detail_df['ProductID'])
            new_products = purchased_products - base_contract_products

            if not new_products:
                st.warning("No new products found.")
            else:
                # If new products exist, proceed with the next steps
                new_products_df = order_detail_df[order_detail_df['ProductID'].isin(new_products)]

                # ---------------------------
                # Prepare Product Data
                # ---------------------------
                product_customer_prices = new_products_df.groupby(
                    ['ProductID', 'Customer-CustomerType']
                )['Price'].max().reset_index()

                product_data = {}
                for product_id in new_products:
                    df_product = product_customer_prices[product_customer_prices['ProductID'] == product_id]
                    customer_types = df_product['Customer-CustomerType'].unique()
                    highest_prices = df_product.set_index('Customer-CustomerType')['Price'].to_dict()
                    product_data[product_id] = {
                        'customer_types': customer_types,
                        'highest_prices': highest_prices
                    }

                # ---------------------------
                # Define Scenario Logic
                # ---------------------------
                def get_actions(product_id, customer_types, highest_prices):
                    actions = {}
                    customer_types_set = set(customer_types)
                    all_types = {'BUILDING', 'HOTEL', 'BUSINESS', 'GYM'}

                    # Only process new products
                    # We will assign prices only for new products and not update existing ones

                    # Scenario 1: Building Only
                    if customer_types_set == {'BUILDING'}:
                        price = highest_prices['BUILDING']
                        actions = {
                            'BUILDINGS': price,
                            'HOTEL': price,
                            'BUSINESS': price,
                            'GYM': price
                        }
                    # Scenario 2: Hotel Only
                    elif customer_types_set == {'HOTEL'}:
                        price = highest_prices['HOTEL']
                        actions = {
                            'HOTEL': price,
                            'BUSINESS': price,
                            'BUILDINGS': price,
                            'GYM': price
                        }
                    # Scenario 3: Business Only
                    elif customer_types_set == {'BUSINESS'}:
                        price = highest_prices['BUSINESS']
                        actions = {
                            'BUSINESS': price,
                            'HOTEL': price,
                            'BUILDINGS': price,
                            'GYM': price
                        }
                    # Scenario 4: Gym Only
                    elif customer_types_set == {'GYM'}:
                        price = highest_prices['GYM']
                        actions = {
                            'GYM': price,
                            'HOTEL': price,
                            'BUILDINGS': price,
                            'BUSINESS': price
                        }
                    # Scenario 5: Building & Hotel Only
                    elif customer_types_set == {'BUILDING', 'HOTEL'}:
                        price_building = highest_prices['BUILDING']
                        price_hotel = highest_prices['HOTEL']
                        actions = {
                            'BUILDINGS': price_building,
                            'HOTEL': price_hotel,
                            'BUSINESS': price_hotel,
                            'GYM': price_hotel
                        }
                    # Scenario 6: Building & Gym
                    elif customer_types_set == {'BUILDING', 'GYM'}:
                        price_building = highest_prices['BUILDING']
                        price_gym = highest_prices['GYM']
                        actions = {
                            'BUILDINGS': price_building,
                            'GYM': price_gym,
                            'HOTEL': price_building,
                            'BUSINESS': price_building
                        }
                    # Scenario 7: Building & Business
                    elif customer_types_set == {'BUILDING', 'BUSINESS'}:
                        price_building = highest_prices['BUILDING']
                        price_business = highest_prices['BUSINESS']
                        actions = {
                            'BUILDINGS': price_building,
                            'BUSINESS': price_business,
                            'HOTEL': price_building,
                            'GYM': price_business
                        }
                    # Scenario 8: Hotel & Gym
                    elif customer_types_set == {'HOTEL', 'GYM'}:
                        price_hotel = highest_prices['HOTEL']
                        price_gym = highest_prices['GYM']
                        actions = {
                            'HOTEL': price_hotel,
                            'GYM': price_gym,
                            'BUILDINGS': price_hotel,
                            'BUSINESS': price_hotel
                        }
                    # Scenario 9: Hotel & Business
                    elif customer_types_set == {'HOTEL', 'BUSINESS'}:
                        price_hotel = highest_prices['HOTEL']
                        price_business = highest_prices['BUSINESS']
                        actions = {
                            'HOTEL': price_hotel,
                            'BUSINESS': price_business,
                            'BUILDINGS': price_hotel,
                            'GYM': price_business
                        }
                    # Scenario 10: Business & Gym
                    elif customer_types_set == {'BUSINESS', 'GYM'}:
                        price_business = highest_prices['BUSINESS']
                        price_gym = highest_prices['GYM']
                        actions = {
                            'BUSINESS': price_business,
                            'GYM': price_gym,
                            'HOTEL': price_business,
                            'BUILDINGS': price_business
                        }
                    # Scenario 11: Building & Hotel & Gym
                    elif customer_types_set == {'BUILDING', 'HOTEL', 'GYM'}:
                        price_building = highest_prices['BUILDING']
                        price_hotel = highest_prices['HOTEL']
                        price_gym = highest_prices['GYM']
                        actions = {
                            'BUILDINGS': price_building,
                            'HOTEL': price_hotel,
                            'GYM': price_gym,
                            'BUSINESS': price_hotel
                        }
                    # Scenario 12: Building & Hotel & Business
                    elif customer_types_set == {'BUILDING', 'HOTEL', 'BUSINESS'}:
                        price_building = highest_prices['BUILDING']
                        price_hotel = highest_prices['HOTEL']
                        price_business = highest_prices['BUSINESS']
                        actions = {
                            'BUILDINGS': price_building,
                            'HOTEL': price_hotel,
                            'BUSINESS': price_business,
                            'GYM': price_business
                        }
                    # Scenario 13: Building & Gym & Business
                    elif customer_types_set == {'BUILDING', 'GYM', 'BUSINESS'}:
                        price_building = highest_prices['BUILDING']
                        price_gym = highest_prices['GYM']
                        price_business = highest_prices['BUSINESS']
                        actions = {
                            'BUILDINGS': price_building,
                            'GYM': price_gym,
                            'BUSINESS': price_business,
                            'HOTEL': price_building
                        }
                    # Scenario 14: Hotel & Gym & Business
                    elif customer_types_set == {'HOTEL', 'GYM', 'BUSINESS'}:
                        price_hotel = highest_prices['HOTEL']
                        price_gym = highest_prices['GYM']
                        price_business = highest_prices['BUSINESS']
                        actions = {
                            'HOTEL': price_hotel,
                            'GYM': price_gym,
                            'BUSINESS': price_business,
                            'BUILDINGS': price_hotel
                        }
                    # Default Case: No specific scenario matches
                    else:
                        print(f"No matching scenario for ProductID: {product_id}, Customer Types: {customer_types_set}")
                        # No actions will be taken for this product

                    return actions

                # ---------------------------
                # Create New Contract Entries
                # ---------------------------
                new_contract_entries = {
                    'BUILDINGS': [],
                    'HOTEL': [],
                    'GYM': [],
                    'BUSINESS': []
                }

                for product_id in product_data:
                    customer_types = product_data[product_id]['customer_types']
                    highest_prices = product_data[product_id]['highest_prices']
                    actions = get_actions(product_id, customer_types, highest_prices)

                    for contract_name, price in actions.items():
                        new_contract_entries[contract_name].append({
                            'ProductID': product_id,
                            'Price': price
                        })

                # Add additional contracts
                new_contract_entries['LIBRARY HOTEL COLLECTION'] = new_contract_entries['HOTEL']
                new_contract_entries['CRUNCH BASE'] = new_contract_entries['GYM']
                new_contract_entries['RETRO FITNESS'] = new_contract_entries['GYM']

                # ---------------------------
                # Export Templates to CSV (In-Memory)
                # ---------------------------
                template_names = [
                    'BUILDINGS', 'BUSINESS', 'CRUNCH BASE', 'GYM', 'HOTEL',
                    'LIBRARY HOTEL COLLECTION', 'RETRO FITNESS'
                ]

                # Prepare output files for download
                output_files = {}
                for contract_name in template_names:
                    entries = new_contract_entries.get(contract_name, [])
                    if entries:
                        df_contract = pd.DataFrame(entries)
                        df_contract.drop_duplicates(subset='ProductID', inplace=True)
                        # No PurchaseCount to merge
                        csv_data = df_contract.to_csv(index=False)
                        output_files[contract_name] = csv_data

                # ---------------------------
                # Generate combined_prices_counts.csv
                # ---------------------------
                # Define the contracts and their corresponding filenames
                contracts = {
                    'GymPrice': 'GYM.csv',
                    'HotelPrice': 'HOTEL.csv',
                    'BuildingPrice': 'BUILDINGS.csv',
                    'BusinessPrice': 'BUSINESS.csv'
                }

                # Define contract to customer type mapping
                contract_customer_mapping = {
                    'GymPrice': ['GYM'],
                    'HotelPrice': ['HOTEL'],
                    'BuildingPrice': ['BUILDING'],
                    'BusinessPrice': ['BUSINESS']
                }

                # Calculate Purchase Counts per Contract
                purchase_counts = {}

                for contract, customer_types in contract_customer_mapping.items():
                    # Filter orders for the current contract's customer types
                    filtered_orders = order_detail_df[order_detail_df['Customer-CustomerType'].isin(customer_types)]
                    
                    # Group by 'ProductID' and count the number of purchases
                    counts = filtered_orders.groupby('ProductID').size().reset_index(name=contract.replace('Price', 'Count'))
                    
                    purchase_counts[contract.replace('Price', 'Count')] = counts

                # Initialize an empty DataFrame to store the combined data
                combined_df = pd.DataFrame()

                # Process Each Contract: Merge Price and PurchaseCount
                for price_column, filename in contracts.items():
                    # Retrieve the corresponding contract CSV data from output_files
                    contract_csv_key = filename.replace('.csv', '').upper()
                    if contract_csv_key in output_files:
                        try:
                            df = pd.read_csv(BytesIO(output_files[contract_csv_key].encode()))
                        except Exception as e:
                            st.error(f"Error reading contract data for {contract_csv_key}: {e}")
                            continue

                        # Keep only 'ProductID' and 'Price' columns
                        if 'Price' not in df.columns:
                            st.warning(f"'Price' column not found in {filename}. Skipping this contract.")
                            continue

                        df = df[['ProductID', 'Price']]

                        # Rename 'Price' column to the contract-specific price column
                        df.rename(columns={'Price': price_column}, inplace=True)

                        # Merge with PurchaseCount
                        count_column = price_column.replace('Price', 'Count')
                        if count_column in purchase_counts:
                            df_count = purchase_counts[count_column]
                            df = df.merge(df_count, on='ProductID', how='left')
                        else:
                            # If no purchase counts calculated, set count to 0
                            df[count_column] = 0

                        # Rename 'Count' column appropriately if it exists
                        if count_column in df.columns:
                            df[count_column] = df[count_column].fillna(0).astype(int)
                        else:
                            df[count_column] = 0

                        # Merge into the combined DataFrame
                        if combined_df.empty:
                            combined_df = df
                        else:
                            combined_df = pd.merge(combined_df, df, on='ProductID', how='outer')
                    else:
                        st.warning(f"Contract data for {contract_csv_key} not found. Skipping.")
                        # Add Price and Count columns with default values
                        combined_df[price_column] = 0
                        count_column = price_column.replace('Price', 'Count')
                        combined_df[count_column] = 0

                # Reorder Columns: Prices First, Then Counts
                price_columns_order = list(contracts.keys())
                count_columns_order = [col.replace('Price', 'Count') for col in contracts.keys()]
                ordered_columns = ['ProductID'] + price_columns_order + count_columns_order

                # If there are additional columns (from merges), include them
                for col in combined_df.columns:
                    if col not in ordered_columns:
                        ordered_columns.append(col)

                # Reorder the DataFrame
                combined_df = combined_df[ordered_columns]

                # Handle Missing Values

                # Replace NaN with 0 for Price columns
                for price_col in price_columns_order:
                    if price_col in combined_df.columns:
                        combined_df[price_col] = combined_df[price_col].fillna(0)
                    else:
                        # If the price column was missing in some contracts, add it with 0
                        combined_df[price_col] = 0

                # Replace NaN with 0 for Count columns and ensure integer type
                for count_col in count_columns_order:
                    if count_col in combined_df.columns:
                        combined_df[count_col] = combined_df[count_col].fillna(0).astype(int)
                    else:
                        # If the count column was missing in some contracts, add it with 0
                        combined_df[count_col] = 0

                # Save the Combined DataFrame to CSV (In-Memory)
                combined_csv = combined_df.to_csv(index=False)
                output_files['COMBINED_PRICES_COUNTS'] = combined_csv

                # ---------------------------
                # Create a ZIP file containing all the CSVs
                # ---------------------------
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for contract_name, csv_data in output_files.items():
                        # Define the filename in the ZIP
                        zip_filename = f"{contract_name.replace(' ', '_').upper()}.csv"
                        zf.writestr(zip_filename, csv_data)

                # Set the download link for the ZIP file
                st.success("Processing completed! You can now download all files as a ZIP.")

                # Download button for the ZIP file
                st.download_button(
                    label="Download All Files as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="contract_templates.zip",
                    mime="application/zip"
                )

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
