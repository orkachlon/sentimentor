# interactive-ml-project-1
First project in the course 'Machine learning as a tool for interactive products'

Files
-----

`dev/nlp_module.py`: the main module to classify and extract features from a given review<br>
`dev/ReviewGenerator.py`: a generator for reviews and their scores<br>
`dev/textVectorization/text_vectorization.py`: library containing a couple of different text vectorizers
- dependencies: `gensim`
- recommended: `cython` - needed for faster training speed

`dev/textVectorization/RakeExtractor.py`: An attempt at the RAKE algorithm using nltk
`dev/dataParsing/data_organizer.py`: An organizer for the 8mil reviews file
- dependencies: `FileData`, `Parser`

`dev/reviewSentiment/SentimentAnalyzer`: a place to try out sentiment analysis using the different vectorization
                                         models<br>
`app/Feature.java`: Represents a feature in a review and its synonyms<br>
`FeatureManager.java`: A manager class for the found features in a review<br>
`SenimentAnalyzer.java`: The module in charge of using the `nlp_module.py` in the Processing app<br>
`app.pde`: The main Processing app that runs the GUI and uses the `nlp_module.py` to process reviews<br>

Description
-----------

This research project attempts to test a machine's ablity to recognize and take part in textual sentimentality.

TODOS
-----

* [ ] add binary arg to parser to specify if the input file should be read in binary mode
* [X] test combined classifier on bigger data to see if better than flair alone
* [X] fix relative paths to be able to run code from nlp_module.py
* [X] add variance to returned synonyms - choose random word instead of taking top one perhaps?
* [X] related to this~^ make synonyms unique? link each feature to the instance in the text?

LINKS
-----

1. Possible entire implementation of classifier including access to influential words:
https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386

2. Amazon 8 million review data set:
http://snap.stanford.edu/data/web-Movies.html

3. Multiclass classification with sklearn, thank you Yuval:
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

4. Ways to vectorize document given a word vectorizer:
https://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence

5. Dealing with imbalanced data:
https://elitedatascience.com/imbalanced-classes

6. Simple kNN implementation:
https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

7. Pre-trained networks:
https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c

8. List of positive and negative words. Used as vocabulary for TfidfVectorizer:
https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

9. How to run commands on terminal from Java:
https://stackoverflow.com/questions/15464111/run-cmd-commands-through-java

Folder structure
----------------

```
interactive-ml-project-1: (root)
├── README.md
├── app: (frontend)
│   ├── Feature.java
│   ├── FeatureManager.java
│   ├── SentimentAnalyzer.java
│   ├── app.pde
│   └── out: (Processing folder)
└── dev: (backend)
    │
    ├── ReviewGenerator.py
    ├── nlp_module.py
    ├── utils.py
    │
    │   # data
    ├── ml-dataset: (raw data files)
    │   ├── bin_test.csv
    │   ├── bin_train.csv
    │   ├── file_info.txt
    │   ├── movies.txt.gz
    │   ├── parser_progress.txt
    │   └── vocab: (word lists for tfidf vocabulary)
    │       ├── neg.txt
    │       └── pos.txt
    ├── csv: (data organized as csv without encoding problems)
    │   ├── bin_test_1.csv
    │   ├── bin_train_1.csv
    │   └── movies_{1-80}.csv
    ├── reviewsByStarRating (the same data from files movies_{1-80}.csv but sorted by rating)
    │   └── movies_{1-80}_score{1-5}.csv
    ├── test: (temp folder for testing)
    │
    │   # packages
    ├── preprocessing: (anything that has to do with cleaning and reorganizing raw data)
    │   ├── FileData.py (not used anymore)
    │   ├── Parser.py
    │   └── preprocessor.py
    ├── reviewSentiment: (anything that has to do with text classification and sentiment analysis)
    │   └── SentimentAnalyzer.py
    └── textVectorization: (anything that has to do with embedding text as vectors)
        ├── RakeExtractor.py (not used anymore)
        ├── text_vectorization.py
        └── textVectorizationModels: (saved vectorizer models)
            ├── csv_80.model
            ├── csv_80.model.trainables.syn1neg.npy
            ├── csv_80.model.wv.vectors.npy
            ├── t2v_50k.model
            ├── t2v_all_lim5k.model
            ├── text8.model
            ├── text8.model.trainables.syn1neg.npy
            ├── text8.model.wv.vectors.npy
            ├── word2vec-google-news-300.model
            └── word2vec-google-news-300.model.vectors.npy
```


Notes
-----

* Not sure if this is needed but I wrote some code that uses pre-trained models from 'gensim' to turn words into
  vectors using the Word2Vec algorithm. This might help since, using this technique, similar words are mapped to
  similar vectors. That makes it easier make good context of words and sentences.

* I might need to map whole reviews into vectors as well.

* Tried method in link no. 1 but it doesn't work well on multiclass, didn't try binary classification.
  tfidf vectorizer kinda works, might need to use it in combination with w2v

* tried 2 more methods for dealing with imbalanced data from link 5, best accuracy i got so far is 50%

* if i want to rid the text of spelling mistakes and stopwords im going to need a lot more processing power


### from 21.04.20

- make classifier more basic - binary or even pre trained
- concentrate on making a working version of the entire process
- get couple hundred relevant words to replace and use them as vocabulary
- till sunday present something that kinda works

### Program flow

```
1. write review -> 2. submit -> 3. predict score onto score slider -> 4. use slider to change sentiment ->
5. refresh to start again
```
1. pretty much done, just need to make design choices
2. press enter or 'submit' button
3. make cmd interface for the classifier and java wrapper class, create slider
4. get synonyms for each word in vocab to replace in review or use Word2Vec to get synonyms
5. make refresh button


### API
```
dev/nlp_module.py:
    Usage: nlp_module.py <review: str>
    Output:  ['POSITIVE'/'NEGATIVE', <confidence>]
             [{syn1, syn2, syn3, syn4},...,{syn1, syn2, syn3, syn4}]
```
