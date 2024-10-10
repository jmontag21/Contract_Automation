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
        df = pd.read_csv(uploaded_file)
        uploaded_data[uploaded_file.name] = df
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

            new_contract_entries['LIBRARY HOTEL COLLECTION'] = new_contract_entries['HOTEL']
            new_contract_entries['CRUNCH BASE'] = new_contract_entries['GYM']
            new_contract_entries['RETRO FITNESS'] = new_contract_entries['GYM']

            # ---------------------------
            # Merge Purchase Counts and Export Templates
            # ---------------------------
            output_dir = 'output_non_base'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

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
                    # Removed the following lines related to PurchaseCount
                    # df_contract = df_contract.merge(purchase_counts, on='ProductID', how='left')
                    # df_contract['PurchaseCount'] = df_contract['PurchaseCount'].fillna(0).astype(int)
                    
                    csv_data = df_contract.to_csv(index=False)
                    output_files[contract_name] = csv_data

            # ---------------------------
            # Create a ZIP file containing all the CSVs
            # ---------------------------
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for contract_name, csv_data in output_files.items():
                    zf.writestr(f"{contract_name.replace(' ', '_').upper()}.csv", csv_data)

            # Set the download link for the ZIP file
            st.success("Processing completed! You can now download all files as a ZIP.")

            # Download button for the ZIP file
            st.download_button(
                label="Download All Files as ZIP",
                data=zip_buffer.getvalue(),
                file_name="contract_templates.zip",
                mime="application/zip"
            )
