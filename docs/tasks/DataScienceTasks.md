# Data Science Tasks

This document details the implementation of the data science tasks requested in the hackathon, specifically focused on data exploration, cleaning, and preprocessing of conversation data.

## 1. Data Analysis and Preprocessing - Explore and clean conversation data

### Requirements:

- Parse the dataset from: `data/groups/thisiscere/messages_thisiscere.csv`
- Implement data cleaning operations:
  - Handle missing values
  - Remove irrelevant columns
  - Format timestamps
  - Text cleaning (special characters, standardization)
  - Optional: Implement spam message detection and filtering
- Implement text tokenization:
  - Handle word boundaries
  - Process punctuation and special characters
  - Manage case sensitivity
  - Optional: stop words removal, contraction handling
- Document your implementation decisions

### Implementation:

#### Dataset Parsing
We implemented a robust data loading system in `optimization/pre_grouping_techniques.py` that can handle various CSV file formats. The system automatically detects relevant columns even when they have different names than expected.

```python
def load_messages(self) -> Optional[pd.DataFrame]:
    """Load and preprocess messages from CSV file."""
    try:
        # Read the CSV file and print its columns
        df = pd.read_csv(self.input_file)
        print(f"Loaded {len(df)} messages from {self.input_file}")
        print(f"Available columns: {', '.join(df.columns)}")
        
        # Try to identify key columns by checking all columns
        id_col = None
        text_col = None
        timestamp_col = None
        username_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            # Identify ID column
            if 'id' in col_lower and not any(x in col_lower for x in ['user', 'conv', 'conversation']):
                id_col = col
            # Identify text column
            elif any(x in col_lower for x in ['text', 'content', 'message']):
                text_col = col
            # Identify timestamp column
            elif any(x in col_lower for x in ['time', 'date', 'timestamp']):
                timestamp_col = col
            # Identify username column
            elif any(x in col_lower for x in ['user', 'name', 'author']):
                username_col = col
```

#### Data Cleaning Operations

##### Handling Missing Values
We implemented detection and handling of null values in critical columns:
- Text: replaced with empty strings
- Usernames: replaced with "unknown_user"
- Missing values: default values are generated when necessary

```python
# Data cleaning
df['text'] = df['text'].fillna('').astype(str)
df['username'] = df['username'].fillna('unknown_user')
```

##### Timestamp Formatting
- Robust conversion of timestamps to pandas datetime objects
- Error handling for invalid or non-existent time formats

```python
# Convert timestamps
try:
    df['datetime'] = pd.to_datetime(df['timestamp'])
except:
    print("Warning: Could not parse timestamp column, using current time")
    df['datetime'] = pd.to_datetime(datetime.now())
```

##### Text Cleaning
- Standardization of texts to string format
- Removal of special characters when necessary
- Analysis and normalization of keywords with TF-IDF

#### Spam Detection and Filtering

A complete spam detector has been developed in `src/detectors/spam_detector.py` with the following features:
- Machine learning techniques to classify messages as spam or not
- Implementation of evaluation metrics (precision, recall, F1)
- The system can evaluate different spam detection models

```python
# From test_spam_detector.py
def test_predict(self):
    """Test spam prediction."""
    # Train the model
    self.detector.fit(self.test_texts, self.test_labels)
    
    # Test single text prediction
    prediction = self.detector.predict("Buy now! Special offer!")
    self.assertIn(prediction[0], [0, 1])
    
    # Test multiple texts prediction
    predictions = self.detector.predict(self.test_texts)
    self.assertEqual(len(predictions), len(self.test_texts))
    self.assertTrue(np.all(np.isin(predictions, [0, 1])))
```

#### Text Tokenization

We use `TfidfVectorizer` from scikit-learn for advanced tokenization with the following features:
- Adjustable configurations for:
  - Stop words handling
  - Document frequency limits (min_df, max_df)
  - Maximum number of features
- Tokenization is used for semantic analysis and keyword-based grouping

```python
self.vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    min_df=1,
    max_df=0.95
)

# Used in keyword_based_grouping:
message_texts = messages['text'].fillna('').astype(str).tolist()
doc_term_matrix = self.vectorizer.fit_transform(message_texts)
# Calculate cosine similarity
similarity_matrix = cosine_similarity(doc_term_matrix)
```

This approach handles:
- Word boundaries through the tokenizer
- Punctuation and special characters
- Case sensitivity
- Stop words removal

#### Advanced Grouping Techniques

We have implemented multiple grouping techniques to detect conversations:

##### Time-Based Grouping
- Adaptive time window based on message density
- Automatic parameter adjustment based on data

```python
# Adjust window based on message density
adaptive_window = min(max(avg_time_diff * 2, 5), time_window_minutes)
```

##### Keyword-Based Grouping
- Use of TF-IDF vectorization and cosine similarity
- Configurable threshold to determine group membership

##### Weighted Group Combination
- Weighted voting system to combine different grouping techniques
- Configurable weights for each grouping method

```python
# Weights for different grouping methods
weights = {
    'time_group': 0.6,
    'keyword_group': 0.4
}
```

#### Implementation Decisions

1. **Automatic Column Detection**: We implemented a system that automatically identifies relevant columns regardless of their exact names, increasing robustness.

2. **Adaptive Time Windows**: Instead of using a fixed time window, we implemented an algorithm that adjusts the window size based on message density.

3. **Multiple Techniques Approach**: We combined different approaches (time, keywords) to obtain better results than with a single method.

4. **Robust Error Handling**: We designed the system to degrade gracefully when there are problems, providing default values and continuing processing.

5. **Evaluation Metrics**: We implemented ARI (Adjusted Rand Index) and other metrics to objectively evaluate the quality of the groupings.

#### Data Exploration and Statistics

The system automatically generates descriptive statistics about the data:

```
INFO - Dataset Statistics:
INFO -   total_messages: 67
INFO -   unique_users: 36
INFO -   time_span: 2025-01-14 01:06:16+00:00 to 2025-01-15 13:59:11+00:00
INFO -   avg_message_length: 147.56716417910448
```

### Questions to Consider

#### How would you approach exploring this dataset?
We implemented a progressive approach:
1. Initial analysis of available columns and data types
2. Descriptive statistics (number of messages, unique users, time range)
3. Visualization and grouping to identify patterns

#### What data quality issues would you look for?
Our system detects and handles:
- Missing values in critical columns
- Inconsistencies in time formats
- Differences in column names
- Empty or malformed texts

#### How would you handle edge cases in the text data?
- We implemented text normalization and cleaning
- Special handling for empty tokens
- Cleaning of special characters when necessary

#### What limitations do you see in this dataset and how could it be improved?
- The number of messages is relatively small for some groups
- Could benefit from additional metadata (language, location)
- Greater consistency in column formats

### Conclusion

The current implementation meets all the specific requirements for data analysis and preprocessing. The system is robust against different input formats, properly handles edge cases, and provides detailed analysis of conversation data.

The code has been designed with a focus on robustness, maintainability, and performance, allowing for future extensions and improvements. 