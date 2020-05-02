import sys
import numpy as np

from typing import List, Any
from spellchecker import SpellChecker
from nltk import pos_tag, word_tokenize

from text_vectorization import W2v, T2v
from sentiment_analysis import CombinedClassifier


MODIFIERS = {0: 'terrible', 1: 'excellent'}
ALLOWED_TAGS = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VBG'}


def extract_features(review: str, t2v: T2v, w2v: W2v) -> List[List[str]]:
    """
    Extracts the features from given review and finds 2 synonyms for each feature.
    Feature extraction is done with T2v (tfidf) and the synonyms are found with W2v (Word2Vec).
    If adjective synonyms are available they'll be taken, otherwise any feature is accepted.
    :param review: to extract features from
    :param t2v: TfidfVectorizer as a T2v
    :param w2v: Word2Vec model as a W2v
    :return: List of features and their synonyms.
    """
    # evaluate this review
    feature_scores = t2v.transform([review]).flatten()
    # remove empty values and get the sorting by maximum score first
    indices = np.argsort(feature_scores)[:: -1]
    feature_scores = np.sort(feature_scores)[:: -1]
    nonzero_indices = np.nonzero(feature_scores)[0]
    sorting = indices[nonzero_indices]
    # get the features corresponding to the scores sorted from best to worst
    features = np.array(t2v.get_feature_names())[sorting]

    # return synonyms for each feature
    syns = []
    splchkr = SpellChecker()
    # try to take only adjectives/adverbs
    tag_dict = {w[0]: w[1]
                for w in pos_tag(word_tokenize(review))
                if w[1] in ALLOWED_TAGS and w[0].lower() in features}
    # if none of the features found are in the allowed tags - use them anyway
    if not len(tag_dict):
        tag_dict = {w[0]: w[1]
                    for w in pos_tag(word_tokenize(review))
                    if w[0].lower() in features}
    # find synonyms for each feature
    used_words = set([f.lower() for f in tag_dict.keys()])
    for f in tag_dict.keys():
        # create list to hold feature in the middle and synonyms from each side
        curr = list([0] * (len(MODIFIERS) + 1))
        # capitalization is preserved
        curr[(len(curr) // 2)] = f
        # convert to lower case for convenience
        f = f.lower()
        for i in range(len(MODIFIERS)):
            pos = [f, MODIFIERS[i]]
            neg = [MODIFIERS[len(MODIFIERS) - 1 - i]]
            # get similarity list
            sim_list = list(map(lambda w: w[0], w2v.wv.most_similar_cosmul(positive=pos, negative=neg)))
            # filter list from used words, non-adjectives and correct spelling errors
            sim_list = [splchkr.correction(w[0])
                        for w in pos_tag(sim_list)
                        if w[1] in ALLOWED_TAGS and w[0] not in used_words]
            # take the most similar
            # skip the cell with the feature itself and save with capitalization if needed
            curr[i if i < len(curr) // 2 else i + 1] = sim_list[0] \
                if curr[len(curr) // 2][0].islower() \
                else sim_list[0][0].upper() + sim_list[0][1:]

            used_words.add(sim_list[0])
        syns.append(curr)
    return syns


def combined_classification(review: str) -> List[Any]:
    """
    Calculates the combined classification
    :param review: review to analyze
    :return: List with first value representing the sentiment via string,
    the second value is the score itself
    """
    classifier = CombinedClassifier()
    score = classifier.predict([review])[0]
    return ["POSITIVE" if score > 0 else "NEGATIVE", (score / 2.0) + .5]


def main(review: str) -> None:
    """
    The main function of this program. Calls the classification and feature extraction functions
    :param review: review to analyze
    """
    print(f"Analyzing: {review}")
    w2v = W2v(path="w2v_csv_80.model", load=True)
    t2v = T2v(name="t2v_50k.model")
    print(combined_classification(review))
    print(extract_features(review, t2v, w2v))


if __name__ == '__main__':
    if len(sys.argv) != 2 or len(sys.argv[1]) == 0:
        print("Usage: nlp_module.py <review: str>")
        exit(-1)
    main(sys.argv[1])


