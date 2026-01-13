# 📺 YouTube Spam Detection

📊 **A machine learning project** focused on detecting spam comments from YouTube videos. The dataset includes comments from five popular videos and aims to classify comments as either spam (1) or not spam (0) using natural language processing (NLP) techniques with machine learning.

**Dataset Source**: [UCI Machine Learning Repository - YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection)

---

## 📁 Project Structure

* `YouTube_Spam_Detection.py`: Python script containing data preprocessing, analysis, and model implementation.
* `YoutubeSpamCollection/`: Folder containing five individual CSV files with labeled YouTube comments.

---

## 📝 Dataset Overview

The dataset consists of YouTube comments from five videos, labeled as spam (`1`) or not spam (`0`).
- **Columns**:
  - `CONTENT`: Text content of the comment.
  - `CLASS`: Label indicating whether the comment is spam (1) or not spam (0).
- **Additional Features**: A new feature `TEXT LENGTH` is engineered to capture the number of characters in each comment.

---

## 📈 Key Insights

### Exploratory Data Analysis:
1. **Text Characteristics**:
   - **Text Length**: Analyzed the distribution of comment lengths for spam and non-spam comments using histograms and boxplots.
   - Visualized class distributions with count plots.
2. **Correlation**:
   - Calculated correlations between `TEXT LENGTH` and `CLASS`.
   - Created a heatmap for better visualization of feature relationships.

### Text Processing:
1. **Text Cleaning**:
   - Removed punctuation.
   - Filtered out common stopwords, retaining user-centric words like "check," "my," etc., to preserve meaningful patterns.
2. **Vectorization**:
   - **TF-IDF** was used to convert processed text into numerical features for machine learning.

### Modeling:
1. **Model 1**: Manual implementation of a Naive Bayes classifier.
2. **Model 2**: A streamlined pipeline with:
   - Count Vectorization
   - TF-IDF Transformation
   - Naive Bayes Classification
3. **Performance**:
   - Confusion matrix and classification report were used to evaluate precision, recall, and accuracy.

---

## 🔍 Key Steps

### 1. Data Cleaning
- Loaded data from five separate files and merged them into a single dataframe.
- Engineered new features and handled missing values.

### 2. Exploratory Data Analysis
- Visualized spam vs. non-spam distributions using histograms, count plots, and boxplots.
- Measured relationships between features with correlation matrices and heatmaps.

### 3. Text Preprocessing
- Cleaned text data by converting to lowercase, removing stopwords, and eliminating punctuation.
- Processed content saved as `PROCESSED` column in the dataframe.

### 4. Feature Engineering
- **TF-IDF** (Term Frequency - Inverse Document Frequency) was used to capture word importance in the dataset.
- Divided the data into training and testing sets.

### 5. Modeling
- **Naive Bayes Classifier**:
  - Built manually and with a pipeline for streamlined workflows.
  - Achieved high accuracy in spam comment classification.

---

## 🛠️ Tools and Libraries Used

* **Python**: Base language for all tasks.
* **Libraries**:
  - `pandas` and `numpy` for data manipulation.
  - `seaborn` and `matplotlib` for data visualization.
  - `nltk` for text preprocessing.
  - `scikit-learn` for feature extraction, modeling, and evaluation.

---

## 🗂️ Key Results

1. **Manual Naive Bayes Implementation**:
   - **Confusion Matrix**:
     ```
     [[442   7]
      [ 19  96]]
     ```
   - **Classification Report**:
     - Precision, recall, and F1-score: High accuracy with minimal misclassifications.

2. **Pipeline Implementation**:
   - Improved efficiency with similar performance to the manual implementation.

---

## 📫 Contact

**LinkedIn**: [www.linkedin.com/in/kanishkayadvv](https://www.linkedin.com/in/kanishkayadvv)  
**Author**: Kanishka Yadav
