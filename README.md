# interactive-ml-project-1
First project in the course 'Machine learning as a tool for interactive products'

Description
-----------

This research project attempts to test a machine's ability to recognize and take part in textual 
sentimentality.

Installation
------------

### Dependencies
#### OS
This program currently only runs on Windows.
#### Anaconda
An installation of [Anaconda](https://www.anaconda.com/products/individual) 
is needed with python 3.7 at least, and the `conda` command should be recognized from cmd.exe. 
Also, a conda environment named `"ml-as-tool-project-1"` should be ready with
the following packages installed:
- [gensim](https://radimrehurek.com/gensim/)
- [flair](https://github.com/flairNLP/flair)
- [sklearn](https://scikit-learn.org/stable/)
- [nltk](https://www.nltk.org/)
- [textblob](https://textblob.readthedocs.io/en/dev/)
- [numpy](https://numpy.org/)
- [spellchecker](https://github.com/barrust/pyspellchecker)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

#### Processing
This app requires an installation of [Processing 3.5.4](https://processing.org/) and the 
[ControlP5](http://www.sojamo.de/libraries/controlP5/) library

### Additional instructions
The trained vectorizers' files are too big to add to this repository so you will need to download
them from my [google drive](
https://drive.google.com/drive/folders/1TCdIGDfix0OMFAbUtn0UaX1T3gYs4jDA?usp=sharing)
and place the assets folder as shown in the Directory tree below.

#### Finally run `app.pde` to run the program!

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
