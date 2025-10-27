import pandas as pd
import sqlite3
import os

# Script to create a SQLite database from multiple CSV files.
DATABASE_FILE = 'olist_ecommerce.db'

# Key is the name of CSV file and value is the name of the table in SQLite.
CSV_FILES_TO_TABLES = {
    'olist_customers_dataset.csv': 'customers',
    'olist_geolocation_dataset.csv': 'geolocation',
    'olist_order_items_dataset.csv': 'order_items',
    'olist_order_payments_dataset.csv': 'order_payments',
    'olist_order_reviews_dataset.csv': 'order_reviews',
    'olist_orders_dataset.csv': 'orders',
    'olist_products_dataset.csv': 'products',
    'olist_sellers_dataset.csv': 'sellers',
    'product_category_name_translation.csv': 'category_translation'
}

# Function to import a CSV file into a SQLite table
def import_csv_to_sqlite(csv_name, table_name, connection):
    """Reads a CSV file and imports it into a SQLite table."""
    try:
        if not os.path.exists(csv_name):
            print(f"⚠️ Warning: File '{csv_name}' not found. Skipping '{table_name}'.")
            return

        df = pd.read_csv(csv_name)
        
        # if_exists='replace' ensures the table is replaced if it already exists
        df.to_sql(
            name=table_name,
            con=connection,
            if_exists='replace',
            index=False # Do not write DataFrame index as a column
        )
        print(f"  ✅ Imported: '{csv_name}' -> Table '{table_name}' ({len(df):,} rows).")

    except Exception as e:
        print(f"  ❌ Error importing file '{csv_name}' to table '{table_name}': {e}")


def main():
    print(f"Starting database creation '{DATABASE_FILE}'...")
    
    # Connect to SQLite database (it will be created if it doesn't exist)
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        
        # Loop over each CSV file and import it into the corresponding table
        for csv_file, table_name in CSV_FILES_TO_TABLES.items():
            import_csv_to_sqlite(csv_file, table_name, conn)
        
        # Confirm all changes are saved
        conn.commit()
        print(f"Database '{DATABASE_FILE}' was created successfully.")

    except ImportError:
        print("\n❌ FATAL ERROR: pandas library is not installed.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR in main process {e}")

    finally:
        if conn:
            conn.close()
            print("SQLite connection closed.")

if __name__ == "__main__":
    main()