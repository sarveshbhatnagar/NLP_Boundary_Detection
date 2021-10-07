import argparse
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import multiprocessing
import pickle

# Question 1
# Would we count " as a word?
# Question 2
# how to handle school...


class WordsFeature:
    def __init__(self, df, label_encoder):
        super().__init__()
        self.df = df
        self.label_encoder = label_encoder
        # self.train_le()

    def get_next_word(self, index, orignal=False):
        """
        Given an index of a word, returns the next word from the dataframe
        params:
            index: int
        returns:
            word: str
        """
        try:
            if orignal:
                return self.df.iloc[index + 1][1]
            return self.label_encoder.transform([self.df.iloc[index + 1][1]])[0]
        except IndexError:
            if orignal:
                return "<END>"
            return self.label_encoder.transform(["<END>"])[0]
        except ValueError:
            # Returning -1 for unseen words
            return -1

    def get_prev_word(self, index, orignal=False):
        """
        Given an index of a word, returns the word before '.' from the dataframe
        params:
            index: int
        returns:
            word: str
        """
        try:

            word = self.df.iloc[index][1]
            if word[-1] == ".":
                if orignal:
                    return word[:-1]
                return self.label_encoder.transform([word[:-1]])[0]
            else:
                # NOT A PERIOD
                # I think it would be better to return a <NAP> token
                # This might also help in cleaning the data
                # If orignal is true return word as is...
                if orignal:
                    return word
                return self.label_encoder.transform(["<NAP>"])[0]
        except ValueError:
            # Returning -1 for unseen words
            return -1
        except IndexError:
            if orignal:
                return "<START>"
            return self.label_encoder.transform(["<START>"])[0]

    def lt_3(self, index):
        """
        Given an index of a word, returns True if length of word before '.' is < 3 in the dataframe
        params:
            index: int
        returns:
            word: str
        """
        word = self.get_prev_word(index, orignal=True)
        return len(word) < 3

    def is_cap_word(self, word):
        """
        Given an index of a word, returns True if the word is capitalized in the dataframe
        params:
            index: int (index of word)
            i: int
        returns:
            is_capital: bool
        """
        try:
            return word[0].isupper()
        except:
            return False

    def actual_len(self, index):
        """
        Returns the actual length of the previous word
        """
        word = self.get_prev_word(index, orignal=True)
        return len(word)

    def get_average_len(self, index):
        """
        Returns the average length of the previous word and next word
        """
        prev_word = self.get_prev_word(index, orignal=True)
        next_word = self.get_next_word(index, orignal=True)
        return (len(prev_word) + len(next_word)) / 2

    def get_avg_len(self, index, window=4):
        """
        Given a window, returns the average length of the words in the window
        (before current word only)

        params:
            index: int
            window: int (default 4)
        returns:
            avg_len: float
        """
        if index < 4:
            words = [len(self.get_prev_word(i, orignal=True)) for i in range(1, index)]
        else:
            words = [
                len(self.get_prev_word(index - i, orignal=True)) for i in range(window)
            ]
        return sum(words) / len(words)

    def get_feature(self, index):
        """
        Given an index, return corresponding features
        params:
            index: int
        returns:
            feature: list (prev word, next word, left_word<3, left_is_cap, right_is_cap, label)
        """
        if index % 1000 == 0:
            print(index)
        feature = []
        # word to the left
        feature.append(self.get_prev_word(index))
        # word to the right
        feature.append(self.get_next_word(index))
        # length of word to the left > 3
        feature.append(self.lt_3(index))
        # Left word is capitalized
        feature.append(self.is_cap_word(self.get_prev_word(index, orignal=True)))
        # Right word is capitalized
        feature.append(self.is_cap_word(self.get_next_word(index, orignal=True)))

        # More three features
        feature.append(self.get_avg_len(index))
        feature.append(self.get_average_len(index))
        feature.append(self.actual_len(index))

        # Finally add label token
        feature.append(self.df[2][index])
        return feature

    def train_le(self):
        """
        It trains label encoder with the appropriate data.
        """

        lisa = [self.get_prev_word(i, orignal=True) for i in range(len(self.df))]
        lisb = [self.get_next_word(i, orignal=True) for i in range(len(self.df))]
        lis = lisa + lisb
        lis.append("<NAP>")
        lis.append("<START>")
        return self.label_encoder.fit(lis)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sentence Boundary Detection")
    parser.add_argument("train", help="Training file")
    parser.add_argument("test", help="Test file")

    args = parser.parse_args()

    # train_file = "SBD.train"
    train_file = args.train
    test_file = args.test

    # Check if train file and test file exists
    # if not os.path.isfile(train_file) or os.path.isfile(test_file):
    #     print("Training or Testing file does not exist")
    #     sys.exit(1)

    # Read the training file
    f = pd.read_csv(train_file, sep=r"\s", header=None)
    f = f.drop(0, axis=1)

    # Resultant file
    #        1    2
    # 0     On  TOK
    # 1   June  TOK
    # 2      4  TOK
    # 3      ,  TOK
    # 4  after  TOK

    wf = WordsFeature(f, label_encoder=LabelEncoder())
    le = wf.train_le()
    ilst = [i for i in range(len(f))]
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=4)
    lst = pool.map(wf.get_feature, ilst)
    pool.close()

    # We now have list containing features. We will now convert this to pandas dataframe.

    # Two methods to move forward, either simply discard TOK and move with just EOS and NEOS
    # or we can keep TOK and move with EOS and NEOS
    # We will use the second method, if it works then good else we will try the first method.

    # Try 1
    # keep features only
    x = [lst[i][:-1] for i in range(len(lst))]
    y = [lst[i][-1] for i in range(len(lst))]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    with open("dtree.pkl", "wb") as dfile:
        pickle.dump(clf, dfile)

    # td = pd.read_csv(test_file, sep=r"\s", header=None)
    # td = td.drop(0, axis=1)
    # twf = WordsFeature(td, le)

    # itlst = [i for i in range(len(td))]
    # tlst = [twf.get_feature(i) for i in range(len(td))]
    # xt = [tlst[i][:-1] for i in range(len(tlst))]
    # yt = [tlst[i][-1] for i in range(len(tlst))]

    # print(clf.predict(xt[:10]))
