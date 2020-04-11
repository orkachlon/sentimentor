import nltk
from operator import itemgetter
from string import punctuation


def is_punct(word):
    return len(word) == 1 and word in punctuation


def is_numeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False


class RakeExtractor:
    def __init__(self):
        self.__stop_words = set(nltk.corpus.stopwords.words())
        self.__top_fraction = 1

    def _generate_candidates(self, sentences):
        phrase_list = []
        words = []
        for sentence in sentences:
            words = map(lambda w: '|' if w in self.__stop_words else w, nltk.word_tokenize(sentence.lower()))
        phrase = []
        for word in words:
            if word == '|' or is_punct(word):
                if len(phrase) > 0:
                    phrase_list.append(phrase)
                    phrase = []
            else:
                phrase.append(word)
        return phrase_list

    @staticmethod
    def _calculate_word_scores(phrase_list):
        word_freq = nltk.FreqDist()
        word_deg = nltk.FreqDist()
        for phrase in phrase_list:
            deg = len([w for w in phrase if not is_numeric(w)]) - 1
            for word in phrase:
                word_freq[word] += 1
                word_deg[word] += deg
        for word in word_freq.keys():
            word_deg[word] += word_freq[word]
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_deg[word] / word_freq[word]
        return word_scores

    @staticmethod
    def _calculate_phrase_scores(phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores[word]
            phrase_scores[' '.join(phrase)] = phrase_score
        return phrase_scores

    def extract(self, text, incl_scores=False):
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidates(sentences)
        word_scores = RakeExtractor._calculate_word_scores(phrase_list)
        phrase_scores = RakeExtractor._calculate_phrase_scores(phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(), key=itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        if incl_scores:
            return sorted_phrase_scores[0:int(n_phrases / self.__top_fraction)]
        else:
            return [x[0] for x in sorted_phrase_scores[: int(n_phrases / self.__top_fraction)]]


if __name__ == '__main__':
    rake = RakeExtractor()
    kw = rake.extract("The movie seemed to get off to a slow start, but I think this is primarily because I couldn't get into the introductory scenes in which greedy scumbags demand their entitlement to an inheritance. Once the plot is set in the introductory scenes, the movie seems to move right along and features excellent performances by the primary cast. I think this is an excellent movie for the entire family, although parents may want to preview the movie to ensure all scenes are appropriate for their children.", incl_scores=True)
    print(kw)
    # nltk.download('punkt')
