
import sys
import numpy as np
from typing import List
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import Laplace, Vocabulary
from nltk.tokenize import sent_tokenize, RegexpTokenizer

NGRAM = 2
RANDOM_SEED = 3

class NgramModel:
    
    def __init__(self, author: str) -> None:
        self.author = author
        self.ngram = NGRAM
        self.raw_sentences = self.split_sentences()
        self.scores = []

    def split_sentences(self) -> List[str]:
        if self.author.endswith("_utf8.txt"):
            self.encoding = "utf-8"
        else:
            self.encoding = "ascii"
        
        with open(f"ngram_authorship_train/{self.author}", "r", encoding=self.encoding) as f:
            return sent_tokenize(f.read())
        
    def split_train_dev(self):
        np.random.shuffle(self.raw_sentences)
        split_idx = int(len(self.raw_sentences) * 0.9)
        self.train, self.train_vocab = self.train_preprocessing(self.raw_sentences[:split_idx])
        self.dev_raw = self.raw_sentences[split_idx:]
        
    def set_train_all(self):
        self.train, self.train_vocab = self.train_preprocessing(self.raw_sentences)
    
    def train_preprocessing(self, sentences: List[str]):
        tokenizer = RegexpTokenizer(r"\w+")
    
        sentences = list(map(str.lower, sentences))
        sentences = list(map(tokenizer.tokenize, sentences))
        
        result, vocab = padded_everygram_pipeline(NGRAM, sentences)
        vocab = Vocabulary(vocab, unk_cutoff=5)
        
        return result, vocab

    @classmethod
    def test_preprocessing(cls, sentences: List[str]):
        tokenizer = RegexpTokenizer(r"\w+")
    
        sentences = list(map(str.lower, sentences))
        sentences = list(map(tokenizer.tokenize, sentences))
        
        result, vocab = padded_everygram_pipeline(NGRAM, sentences)
        
        return result, vocab

    def training(self):
        self.lm = Laplace(NGRAM)
        self.lm.fit(self.train, self.train_vocab)

    def inference(self, raw_sentences: List[str]):
        dev, _ = NgramModel.test_preprocessing(raw_sentences)
        self.scores.append(list(map(self.lm.perplexity, dev)))
        

def main(
    author_list: List[str],
    test_tf: bool = False,
    test_file: str = ""
) -> None:
    """
    
    """
    
    if not test_tf:
        print("splitting into training and development...")
        for author in author_list:
            author_var = author.replace("_utf8", "").replace(".txt", "")
            
            globals()[author_var] = NgramModel(author)
            globals()[author_var].split_sentences()
            globals()[author_var].split_train_dev()
        
        print("training LMs... (this may take a while)")
        for author in author_list:
            author_var = author.replace("_utf8", "").replace(".txt", "")

            globals()[author_var].training()
            for author in author_list:
                author_var_for_inference = author.replace("_utf8", "").replace(".txt", "")
                globals()[author_var].inference(globals()[author_var_for_inference].dev_raw)
            
        print("Results on dev set:")
        ngram_list = []
        for author in author_list:
            author_var = author.replace("_utf8", "").replace(".txt", "")
            ngram_list.append(globals()[author_var])
        
        calculate_accuracy(ngram_list)
        
        with open("generated.txt", "w") as f:
            for author in author_list:
                author_var = author.replace("_utf8", "").replace(".txt", "")
                f.write(f"{author_var}:\n")
                for _ in range(5):
                    perplexity, text = generate_text(globals()[author_var])
                    f.write(f"perplexity: {perplexity}, text: {text}\n")
                f.write("\n")
        
    else:
        test_raw_sentences = load_test(test_file)
        
        print("training LMs... (this may take a while)")
        for author in author_list:
            author_var = author.replace("_utf8", "").replace(".txt", "")
            
            globals()[author_var] = NgramModel(author)
            globals()[author_var].set_train_all()
            globals()[author_var].training()
            globals()[author_var].inference(test_raw_sentences)
        
        ngram_list = []
        for author in author_list:
            author_var = author.replace("_utf8", "").replace(".txt", "")
            ngram_list.append(globals()[author_var])
        
        authors = [ngram.author.replace("_utf8", "").replace(".txt", "") for ngram in ngram_list]    
        tmp_score = [ngram.scores[0] for ngram in ngram_list]
        prediction = list(np.argmin(tmp_score, axis=0))
        
        for p in prediction:
            print(f"{authors[p]}")
        

def generate_text(ngram: NgramModel) -> str:
    generated = ngram.lm.generate(20)
    perplexity = ngram.lm.perplexity(generated)
    return perplexity, " ".join(ngram.lm.generate(20))
            
def load_test(test_file: str) -> List[str]:
    with open(test_file, "r") as f:
        test = f.readlines()
    return test
        
def calculate_accuracy(ngram_list: List[NgramModel]) -> None:
    
    authors = [ngram.author.replace("_utf8", "").replace(".txt", "") for ngram in ngram_list]
    labels = [[i for _ in range(len(score))] for i, score in enumerate(ngram_list[0].scores)]
    
    predictions = []
    for idx, author in enumerate(authors):
        tmp_score = [ngram.scores[idx] for ngram in ngram_list]
        predictions.append(list(np.argmin(tmp_score, axis=0)))
    
    comparison = []
    for label, prediction in zip(labels, predictions):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        for l, p in zip(label, prediction):
            if l == p:
                tp += 1
                tn += (len(authors) - 1)
            else:
                tn += (len(authors) - 2)
                fp += 1
                fn += 1
        
        acc = round((tp + tn) / (tp + tn + fp + fn) * 100, 1)
        comparison.append(acc)
            
    for idx, author in enumerate(authors):
        print(f"{author}    {comparison[idx]}% correct")


if __name__ == "__main__":
    authors_path = sys.argv[1]
    with open(authors_path, "r") as f:
        author_list = f.readlines()
        author_list = [author.replace("\n", "") for author in author_list]
    
    test_tf = False
    test_file = ""
    if len(sys.argv) >= 3:
        assert sys.argv[2] == "-test", "Please set arg as -test"
        test_tf = True
        test_file = sys.argv[3]
    
    main(author_list, test_tf=test_tf, test_file=test_file)