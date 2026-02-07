# Sentiment Analysis on Customer Reviews 🎯

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-86.8%25-brightgreen.svg)

An end-to-end Machine Learning project that performs sentiment analysis on customer reviews using Natural Language Processing (NLP) techniques.

## 📊 Project Overview

This project analyzes **113,691 customer reviews** and classifies them as **Positive**, **Negative**, or **Neutral** using multiple Machine Learning algorithms. The best performing model achieved **86.8% accuracy**.

### Key Features
- ✅ Comprehensive text preprocessing pipeline
- ✅ Multiple ML model comparison (4 algorithms)
- ✅ Advanced visualization (word clouds, confusion matrices)
- ✅ Real-time sentiment prediction system
- ✅ Detailed performance metrics and analysis

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Total Reviews Analyzed** | 113,691 |
| **Best Model** | Logistic Regression |
| **Accuracy** | 86.8% |
| **Precision** | 0.73 |
| **Recall** | 0.68 |
| **F1-Score** | 0.71 |

### Sentiment Distribution
- 🟢 **Positive:** 78.3% (89,025 reviews)
- 🔴 **Negative:** 14.2% (16,181 reviews)
- 🟡 **Neutral:** 7.5% (8,485 reviews)

## 🛠️ Tech Stack

- **Programming Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **NLP Libraries:** NLTK, TextBlob
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Development Environment:** Google Colab

## 📁 Project Structure

```
sentiment-analysis-project/
│
├── notebooks/
│   └── Sentiment_Analysis_Final.ipynb    # Main project notebook
│
├── data/
│   └── Reviews.csv                        # Dataset (not included - see instructions)
│
├── models/
│   ├── best_model.pkl                     # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl              # TF-IDF vectorizer
│
├── images/
│   ├── word_clouds.png                    # Sentiment word clouds
│   ├── model_comparison.png               # Model performance comparison
│   ├── confusion_matrix.png               # Confusion matrix visualization
│   └── sentiment_distribution.png         # Dataset distribution
│
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
└── LICENSE                                # MIT License
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Colab (optional, for notebook execution)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

## 📊 Dataset

The project uses a customer reviews dataset with the following structure:
- **Source:** Amazon Fine Food Reviews (Kaggle)
- **Size:** 113,691 reviews
- **Columns:** Text, Score, Sentiment

### Download Dataset
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets).

**Note:** Place the `Reviews.csv` file in the `data/` directory.

## 💻 Usage

### Running the Jupyter Notebook

1. **Open Google Colab or Jupyter Notebook**
```bash
jupyter notebook notebooks/Sentiment_Analysis_Final.ipynb
```

2. **Execute all cells in sequence**

The notebook includes:
- Data loading and exploration
- Text preprocessing
- Feature engineering
- Model training and evaluation
- Visualization generation
- Interactive prediction system

### Using Trained Model for Predictions

```python
import pickle
from text_preprocessing import clean_text, preprocess_text

# Load trained model and vectorizer
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Make prediction
def predict_sentiment(text):
    # Preprocess
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)
    
    # Vectorize
    vectorized = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    
    return prediction

# Example
review = "This product is absolutely amazing! I love it!"
sentiment = predict_sentiment(review)
print(f"Sentiment: {sentiment}")  # Output: Positive
```

## 🔬 Methodology

### 1. Data Preprocessing
- Text cleaning (remove URLs, mentions, special characters)
- Lowercase conversion
- Tokenization
- Stop word removal
- Lemmatization

### 2. Feature Engineering
- **TF-IDF Vectorization**
  - Max features: 5,000
  - N-gram range: (1, 2)
  - Captures word importance and context

### 3. Model Training
Trained and compared 4 algorithms:

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Logistic Regression** | **86.8%** | Best performer ⭐ |
| Naive Bayes | 81.9% | Fast, good baseline |
| Random Forest | - | Ensemble method |
| SVM (Linear) | - | Support vectors |

### 4. Model Evaluation
- Confusion matrix analysis
- Precision, Recall, F1-Score
- Cross-validation
- Stratified sampling for imbalanced dataset

## 📈 Visualizations

The project includes comprehensive visualizations:

### Word Clouds
![Word Clouds](images/word_clouds.png)
*Most frequent words in positive, negative, and neutral reviews*

### Model Comparison
![Model Comparison](images/model_comparison.png)
*Performance comparison across all models*

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)
*Detailed prediction analysis*

### Sentiment Distribution
![Sentiment Distribution](images/sentiment_distribution.png)
*Dataset sentiment breakdown*

## 🎓 Key Learnings

1. **Data Quality is Crucial**
   - Spent 40% of time on data preprocessing
   - Real-world data is messy and requires careful cleaning

2. **Feature Engineering Matters**
   - TF-IDF improved accuracy from 70% to 86.8%
   - Bigrams captured important context

3. **Handling Imbalanced Data**
   - Stratified sampling maintained class distribution
   - Weighted loss functions improved minority class performance

4. **Model Selection**
   - Logistic Regression performed best for this text classification task
   - Simple models can outperform complex ones with proper feature engineering

## 🚧 Challenges & Solutions

### Challenge 1: Imbalanced Dataset
- **Problem:** 78% positive, 14% negative, 7% neutral
- **Solution:** Stratified sampling, weighted loss functions

### Challenge 2: Text Preprocessing
- **Problem:** Messy data with emojis, URLs, special characters
- **Solution:** Comprehensive regex-based cleaning pipeline

### Challenge 3: High Dimensionality
- **Problem:** Large vocabulary creating sparse features
- **Solution:** TF-IDF with max_features=5000

### Challenge 4: Context Understanding
- **Problem:** Phrases like "not bad" being misclassified
- **Solution:** Bigram features to capture word pairs

## 🔮 Future Improvements

- [ ] Implement BERT/Transformers for improved accuracy
- [ ] Add multi-language support
- [ ] Deploy as REST API using Flask/FastAPI
- [ ] Create web interface with Streamlit
- [ ] Handle sarcasm and context better
- [ ] Add aspect-based sentiment analysis
- [ ] Implement real-time streaming analysis

## 📝 Project Report

For a detailed project report including methodology, results, and analysis, see [PROJECT_REPORT.md](PROJECT_REPORT.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**LinkedIn:** https://www.linkedin.com/in/nelluri-chaitrika-sri-nidhi-4319b5329

**Project Link:** [https://github.com/ChaitrikaNelluri/sentiment-analysis-project](https://github.com/YOUR_USERNAME/sentiment-analysis-project)

## 🙏 Acknowledgments

- **Coding Samurai**- For the internship opportunity
- **Kaggle** - For providing the dataset
- **NLTK & Scikit-learn** - For excellent NLP and ML tools
- **Stack Overflow Community** - For troubleshooting help

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

## 📚 Additional Resources

### Tutorials & Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [TextBlob Documentation](https://textblob.readthedocs.io/)

### Related Papers
- "Sentiment Analysis and Opinion Mining" by Bing Liu
- "Natural Language Processing with Python" by Steven Bird

### Useful Links
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Understanding Logistic Regression](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

---

**Made with ❤️ and ☕ by [Your Name]**

*This project was completed as part of the Coding Samurai Internship Program*

---

## 📊 Project Statistics

```
Project Duration: 3 weeks
Lines of Code: ~2,000
Data Processed: 113,691 reviews
Models Trained: 4
Visualizations Created: 15+
Coffee Consumed: ∞ ☕
```

## 🎯 Skills Demonstrated

- ✅ Natural Language Processing (NLP)
- ✅ Machine Learning Model Training
- ✅ Data Preprocessing & Cleaning
- ✅ Feature Engineering (TF-IDF)
- ✅ Model Evaluation & Optimization
- ✅ Data Visualization
- ✅ Python Programming
- ✅ Statistical Analysis
- ✅ Documentation

---

**Last Updated:** February 2026
