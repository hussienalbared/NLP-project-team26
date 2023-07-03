import csv
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


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

# Create a list to store the comment lengths
comment_lengths = []

# Iterate through the preprocessed comments and calculate the lengths
for row in data:
    comment = row[comment_index]
    comment_length = len(comment.split())
    comment_lengths.append(comment_length)

# Plot a histogram of comment lengths
plt.hist(comment_lengths, bins=20, edgecolor='black')
plt.xlabel('Comment Length')
plt.ylabel('Frequency')
plt.title('Distribution of Comment Lengths')
plt.show()


# Combine all the preprocessed comments into a single string
all_comments = ' '.join([row[comment_index] for row in data])

# Split the string into individual words
words = all_comments.split()

# Count the frequency of each word
word_counts = Counter(words)

# Generate a word cloud from the word counts
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words')
plt.show()


# Count the number of sarcastic and non-sarcastic comments
sarcasm_count = 0
non_sarcasm_count = 0
for row in data:
    label = int(row[header.index('label')])
    if label == 1:
        sarcasm_count += 1
    else:
        non_sarcasm_count += 1

# Create a pie chart to visualize the distribution
labels = ['Sarcasm', 'Non-sarcasm']
sizes = [sarcasm_count, non_sarcasm_count]
colors = ['#ff9999', '#66b3ff']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Sarcasm vs. Non-sarcasm')
plt.show()
#########