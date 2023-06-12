#### approach

SVM: using glove embedding as the representation for each word. using NLTK to do lemmatizing and stemming, remove stop words and recover words to its normal form.

SpaCy: used "eng-web-core-md" as the underlying model. Extracting only nouns, verbs, pronoun, adjectives, adverbs, proper noun. So that those stop words won't affect the similarity calculation.

Two ways to calculate similarity, first mean and then calculate cosine similarity. Or comapre similarity with each token, and then mean to get the final point. Using a LinearSVM to determine the boundary for each choices' similarity score.

#### accuracy

|                | no facts | with facts | with facts and common sense |
| -------------- | -------- | ---------- | --------------------------- |
| svm            | 0.288    | 0.304      | 0.26                        |
| svm_with_spacy | 0.25     | 0.342      | 0.332                       |

