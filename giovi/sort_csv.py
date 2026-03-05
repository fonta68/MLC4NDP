import csv
import os
import sys

def sort_csv_by_column_inplace(filename, one_based_column_index):
    # Convert from 1-based to 0-based indexing
    column_index = one_based_column_index - 1
    
    # Read the CSV file
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row
        data_rows = list(reader)  # Get all data rows
    
    # Check if the column index is valid
    if column_index < 0 or column_index >= len(header):
        print(f"Error: Column index {one_based_column_index} is out of range. File has {len(header)} columns.")
        sys.exit(1)
    
    # Sort the data rows by the specified column
    # Convert the value to float for proper numeric sorting
    sorted_rows = sorted(data_rows, key=lambda row: float(row[column_index]))
    
    # Write the sorted data back to the original file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header first
        writer.writerows(sorted_rows)  # Write the sorted data
    
    column_name = header[column_index]
    #print(f"File '{filename}' has been sorted by {column_name} (column {one_based_column_index}) and updated in place.")

def print_usage():
    print("Usage: python script.py <filename> <column_index>")
    print("  - filename: path to the CSV file")
    print("  - column_index: index of the column to sort by (1-based, 1 is the first column)")
    print("Example: python script.py data.csv 4")

if __name__ == "__main__":
    # Check if we have the right number of arguments
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Try to convert the column index to an integer
    try:
        one_based_column_index = int(sys.argv[2])
        if one_based_column_index < 1:
            print("Error: Column index must be at least 1")
            print_usage()
            sys.exit(1)
    except ValueError:
        print("Error: Column index must be an integer")
        print_usage()
        sys.exit(1)
    
    # Check if the file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    # Sort the file
    sort_csv_by_column_inplace(filename, one_based_column_index)
