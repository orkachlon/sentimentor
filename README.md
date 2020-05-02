# interactive-ml-project-1
First project in the course 'Machine learning as a tool for interactive products'

Description
-----------

This research project attempts to test a machine's ability to recognize and take part in textual 
sentimentality.

Installation
------------

### Dependencies

#### Anaconda
An installation of [Anaconda](https://www.anaconda.com/products/individual) 
is assumed to be available at `C:\ProgramData\Anaconda` along 
with a conda environment named `"ml-as-tool-project-1"`.
The following python packages need to be available in the conda env:
- [gensim](https://radimrehurek.com/gensim/)
- [flair](https://github.com/flairNLP/flair)
- [sklearn](https://scikit-learn.org/stable/)
- [nltk](https://www.nltk.org/)
- [textblob](https://textblob.readthedocs.io/en/dev/)
- [numpy](https://numpy.org/)
- [spellchecker](https://github.com/barrust/pyspellchecker)

#### Processing
This app requires an installation of [Processing 3.5.4](https://processing.org/) and the 
[ControlP5](http://www.sojamo.de/libraries/controlP5/) library

Instructions
------------

1. write review
2. submit
3. score is predicted onto slider
4. use slider to change sentiment
5. type another review to start again

Files
-----
### Directory tree
```
root/
├── README.md
├── app
│   ├── Feature.java
│   ├── FeatureManager.java
│   ├── SATester.java
│   ├── SentimentAnalyzer.bat
│   ├── SentimentAnalyzer.java
│   └── app.pde
├── assets
│   ├── csv
│   │   ├── bin_test_1.csv
│   │   ├── bin_train_1.csv
│   │   └── movies_{1-80}.csv
│   ├── raw
│   │   ├── bin_test.csv
│   │   ├── bin_train.csv
│   │   ├── file_info.txt
│   │   ├── movies.txt.gz
│   │   └── parser_progress.txt
│   ├── vectorizers
│   │   ├── t2v_50k.model
│   │   ├── w2v_csv_80.model
│   │   ├── w2v_csv_80.model.trainables.syn1neg.npy
│   │   └── w2v_csv_80.model.wv.vectors.npy
│   └── vocab
│       ├── neg.txt
│       └── pos.txt
└── dev
    ├── Parser.py
    ├── ReviewGenerator.py
    ├── nlp_module.py
    ├── preprocessor.py
    ├── sentiment_analysis.py
    ├── text_vectorization.py
    └── utils.py
```

### File description
`dev/`:
- `nlp_module.py`: the main module to classify and extract features from a given review
- `ReviewGenerator.py`: a generator for reviews and their scores
- `text_vectorization.py`: library containing a couple of different text vectorizers
- `data_organizer.py`: An organizer for the 8mil reviews file
- `SentimentAnalyzer`: a place to try out sentiment analysis using the different vectorization
                         models


`app/`:
- `Feature.java`: Represents a feature in a review and its synonyms
- `FeatureManager.java`: A manager class for the found features in a review
- `SenimentAnalyzer.java`: The module in charge of using the `nlp_module.py` in the Processing app
- `app.pde`: The main Processing app that runs the GUI and uses the `nlp_module.py` to process reviews

References
----------
Not all of these references are used in the project but all of them helped me
learn this topic and develop this project.

- [Possible entire implementation of classifier including access to influential words](
https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386)

- [Amazon 8 million review data set](
http://snap.stanford.edu/data/web-Movies.html)

- [Multiclass classification with sklearn](
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)
 thank you Yuval
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

- [How to run commands on terminal from Java](
https://stackoverflow.com/questions/15464111/run-cmd-commands-through-java)
