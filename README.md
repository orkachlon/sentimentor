# interactive-ml-project-1
First project in the course 'Machine learning as a tool for interactive products'

Description
-----------

In this project I explore the ability of a machine to identify and manipulate textual emotion using various ML models.

The program accepts text containing sentiment, predicts its score onto a slider, and then allows the user to move that slider to gradually increase or decrease the positivity of the text.


Installation
------------
To run the program run `app.pde` in the `app` folder. Please note the dependedncies first.

### Dependencies
#### OS
This program currently runs only on Windows.
#### Anaconda
An installation of [Anaconda](https://www.anaconda.com/products/individual) 
is needed with python 3.7, and the `conda` command should be recognized from cmd.exe. 
Also, a conda environment named `"ml-as-tool-project-1"` should be ready with
the following packages installed:

| package                                                   | version |
|-----------------------------------------------------------|---------|
| [gensim](https://radimrehurek.com/gensim/)                | 3.8.1   |
| [flair](https://github.com/flairNLP/flair)                | 0.4.5   |
| [sklearn](https://scikit-learn.org/stable/)               | 0.22.1  |
| [nltk](https://www.nltk.org/)                             | 3.4.4   |
| [textblob](https://textblob.readthedocs.io/en/dev/)       | 0.15.3  |
| [numpy](https://numpy.org/)                               | 1.18.1  |
| [spellchecker](https://github.com/barrust/pyspellchecker) | 0.5.3   |
| [pandas](https://pandas.pydata.org/)                      | 1.0.3   |
| [matplotlib](https://matplotlib.org/)                     | 3.2.1   |
| [seaborn](https://seaborn.pydata.org/)                    | 0.10.0  |

#### Processing
This app requires an installation of [Processing 3.5.4](https://processing.org/) with [java 8](https://www.oracle.com/java/technologies/javase-jdk8-downloads.html)
and the [ControlP5](http://www.sojamo.de/libraries/controlP5/) library.

#### Additional dependencies
The trained vectorizers' files are too big to add to this repository so you will need to download
them from my [google drive](
https://drive.google.com/drive/folders/1TCdIGDfix0OMFAbUtn0UaX1T3gYs4jDA?usp=sharing)
and place the assets folder as shown in the Directory tree below.

Instructions
------------

1. Write review
2. Submit
3. Score is predicted onto slider
4. Use slider to change sentiment
5. Type another review to start again

Files
-----
### Directory tree
```
interactive-ml-project-1/
├── README.md
├── app
│   ├── Feature.java
│   ├── FeatureManager.java
│   ├── SentimentAnalyzer.bat
│   ├── SentimentAnalyzer.java
│   └── app.pde
├── assets
│   └── vectorizers
│       ├── t2v_50k.model
│       ├── w2v_csv_80.model
│       ├── w2v_csv_80.model.trainables.syn1neg.npy
│       └── w2v_csv_80.model.wv.vectors.npy
└── dev
    ├── Parser.py
    ├── ReviewGenerator.py
    ├── nlp_module.py
    ├── preprocessor.py
    ├── sentiment_analysis.py
    └── text_vectorization.py
```

Implementation details
----------------------
**Classification**: reviews are classified using a weighted sum of vader, flair and textblob text classifiers.

**Feature extraction**: features are extracted using a tf-idf vectorizer trained with a given vocabulary.

**Synonym generation**:  synonyms are generated using gensim's Word2Vec on the found features.

**Word swapping**: a bucket system is created, where each bucket represents a different word swap. Each transition
between buckets swaps a single word.

### File description
`dev/`:
- `Parser.py`: a parser for the Amazon 8M review and the IMDB 50K review datasets
- `ReviewGenerator.py`: generators for reviews and their scores
- `preprocessing.py`: contains functions to preprocess data
- `text_vectorization.py`: library containing two different text vectorizers: Word2Vec and TfidfVectorizer
- `sentiment_analysis.py`: contains the implementation of the classifier used for this project as well as
                           useful functions to test out classifiers
- `nlp_module.py`: the main module to classify and extract features from a given review


`app/`:
- `Feature.java`: Represents a feature in a review and its synonyms
- `FeatureManager.java`: A manager class for the found features in a review
- `SenimentAnalyzer.java`: The module in charge of using the `nlp_module.py` in the Processing app
- `app.pde`: The main Processing app that runs the GUI and uses the `SentimentAnalyzer.java` to process reviews

References
----------
Not all of these references are used in the project but all of them helped me
learn this topic and develop this project.

- [Implementation of classifier including access to influential words](
https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386)

- [Amazon 8 million review data set](
http://snap.stanford.edu/data/web-Movies.html)

- [Multiclass classification with sklearn](
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)
 thank you yuvalpadan
- [Ways to vectorize document given a word vectorizer](
https://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence)

- [Dealing with imbalanced data](
https://elitedatascience.com/imbalanced-classes)

- [Simple kNN implementation](
https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75)

- [Pre-trained networks](
https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c)

- [List of positive and negative words. Used as vocabulary for TfidfVectorizer](
https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)
