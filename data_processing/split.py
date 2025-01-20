# Description: Split the CSV file into training and val datasets
# Run after process.py
import csv

# Parameters
input_csv = r'/Volumes/HP X900W/data-2024.09.csv'
train_csv = r'/Volumes/HP X900W/data_tr.csv'
test_csv = r'/Volumes/HP X900W/data_val.csv'
test_ratio = 0.1 # deprecated
train_num = 320000000 # 320M train positions

# Step 1: Open input and output files
count = 0
with open(input_csv, 'r') as infile, \
     open(train_csv, 'w', newline='') as trainfile, \
     open(test_csv, 'w', newline='') as testfile:

    reader = csv.reader(infile)
    train_writer = csv.writer(trainfile)
    test_writer = csv.writer(testfile)

    # Step 2: Handle the header (optional)
    header = next(reader)  # Reads the first row as the header
    train_writer.writerow(header)
    test_writer.writerow(header)

    # Step 3: Shuffle and split rows
    for row in reader:
        if row == []: # Skip empty rows
            continue
        count += 1
        if count < train_num:
            train_writer.writerow(row)
        else:
            test_writer.writerow(row)

print(f"Train dataset saved to {train_csv}")
print(f"Test dataset saved to {test_csv}")