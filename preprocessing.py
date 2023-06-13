import csv
import re


def preprocess_comment(comment):
    # Remove punctuation
    comment = re.sub(r'[^\w\s]', '', comment)

    # Convert to lowercase
    comment = comment.lower()

    # Remove stop words manually
    stop_words = ['a', 'an', 'the', 'is', 'are']  # Add or modify stop words as needed
    tokens = comment.split()
    filtered_comment = [word for word in tokens if word not in stop_words or word.isdigit()]

    # Join the preprocessed tokens back into a string
    preprocessed_comment = ' '.join(filtered_comment)

    return preprocessed_comment


# Open the CSV file and load the data into memory
data = []
with open('data/train-balanced-sarcasm.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Assuming the first row contains column headers

    # Find the index of the comment column
    comment_index = header.index('comment')

    # Iterate through the rows and preprocess the comments
    for row in reader:
        comment = row[comment_index]
        preprocessed_comment = preprocess_comment(comment)

        # Update the comment column with the preprocessed comment
        row[comment_index] = preprocessed_comment

        # Append the updated row to the data list
        data.append(row)

# Write the preprocessed data to a new CSV file
with open('preprocessed_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)