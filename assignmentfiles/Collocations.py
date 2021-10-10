import re
from collections import Counter
import math
import warnings
import argparse

warnings.filterwarnings("ignore")


class NLPCollocations:
    def __init__(self, data):
        self.data = data
        self.tokens = self.split_data()
        self.tokens = self.remove_stopwords(self.tokens)
        self.unigrams = self.get_unigrams(self.tokens)
        self.bigrams = self.get_bigrams(self.tokens)
        self.bigram_freq = self.get_bigram_freq(self.bigrams)

    def split_data(self):
        """
        Splits the data based on non alpha numeric tokens. Used this instead of \W+ or something
        because W+ considers '.' too. We should not split by . as it will remove terms like U.S.

        params:
            data: text
        returns:
            list of tokens
        """
        rle = re.split("[\-\" `'\\n:,\\t#\$%&\(\)\*]+", self.data)
        return rle

    def remove_stopwords(self, tokens):
        """
        Removes stopwords from the data.

        params:
            data: tokens
        returns:
            list of tokens
        """
        stopwords = [
            "the",
            "of",
            "in",
            "a",
            "this",
            "them",
            "their",
            "he",
            "she",
            "if",
            "it",
            "is",
            "n",
            "to",
            "t",
            "and",
            "s",
            "wo",
        ]
        wle = []
        for word in tokens:
            if len(word) == 0:
                continue
            if word == "." or word == "/":
                continue
            if word == "..":
                continue
            if word == "...":
                continue
            if word[0] == ".":
                continue
            if word.lower() in stopwords:
                continue
            wle.append(word)

        return wle

    def get_unigrams(self, tokens):
        """
        Gets the unigrams from the data.

        params:
            data: text
        returns:
            list of tokens
        """
        return dict(Counter(tokens))

    def get_bigrams(self, tokens):
        """
        Returns bigrams from data.

        params:
            data: list of tokens
        returns:
            list of bigrams (word1,word2)
        """
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))

        return bigrams

    def get_bigram_freq(self, bigrams):
        """
        Returns the frequency of each bigram.

        params:
            bigrams: list of bigrams
        returns:
            dict: freq of bigrams
        """
        return Counter(bigrams)

    def expected_prob(self, word, total):
        """
        Expected probability of the word in corpus.
        """
        return self.unigrams[word] / total

    def chi_square(self, word1, word2):
        """
        Tried using two methods, one from book and other from directly calculating. Both gives same result.
        """
        both_holds = self.bigram_freq[(word1, word2)]
        first_holds = self.unigrams[word1] - self.bigram_freq[(word1, word2)]
        second_holds = self.unigrams[word2] - self.bigram_freq[(word1, word2)]
        none_holds = sum(self.bigram_freq.values()) - self.bigram_freq[(word1, word2)]
        observed = [both_holds, second_holds, first_holds, none_holds]

        total = sum(self.unigrams.values())
        # pf: probability of first word
        # ps : probability of second word
        pf = self.expected_prob(word1, total)
        ps = self.expected_prob(word2, total)
        # first holds, second holds
        first = pf * ps * total
        # first does not hold, second holds
        second = (1 - pf) * ps * total
        # first holds, second does not hold
        third = pf * (1 - ps) * total
        # first does not hold, second does not hold
        fourth = (1 - pf) * (1 - ps) * total

        expected = [first, second, third, fourth]

        chi = 0

        # Note: Tried 2 methods, first from book and second from directly calculating.
        # Both gives same result.
        # Values are coming too large tho, I am not sure why.
        # Question: Is the denominator sum(expected) instead of expected[i]?
        # That returns better results. but well, just implemented the formula.
        for i in range(len(expected)):
            chi += ((observed[i] - expected[i]) ** 2) / expected[i]

        return chi

    def pmi(self, word1, word2):
        """
        Calculates the pointwise mutual information of the two words.
        """
        pw1w2 = self.bigram_freq[(word1, word2)] / sum(self.bigram_freq.values())
        total = sum(self.unigrams.values())

        pw1 = self.expected_prob(word1, total)
        pw2 = self.expected_prob(word2, total)

        return math.log(pw1w2 / (pw1 * pw2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sentence Boundary Detection")
    parser.add_argument("method", help="Specify Choice of Method (chi-square or PMI)")
    args = parser.parse_args()
    method = args.method

    with open("Collocations", "r") as f:
        r = f.read()

    collocs = NLPCollocations(r)

    # Using chi-square
    if method.lower() == "chi-square":
        csli = []
        for bg in collocs.bigram_freq.keys():
            csli.append((bg, collocs.chi_square(bg[0], bg[1])))

        # Sort the bigrams by chi-square value
        sorted_chi = sorted(csli, key=lambda x: x[1], reverse=True)

        print("Top 20 Reverse Sorted based on chi-square")
        print(sorted_chi[:20])

    # Using PMI
    elif method.lower() == "pmi":
        pmili = []
        for bg in collocs.bigram_freq.keys():
            pmili.append((bg, collocs.pmi(bg[0], bg[1])))

        # Sort the bigrams by PMI value
        sorted_pmi = sorted(pmili, key=lambda x: x[1], reverse=True)
        print("Top 20 Reverse Sorted based on PMI")
        print(sorted_pmi[:20])

    else:
        print("Invalid method")
