from ldamallet import LdaMallet
from gensim.corpora import Dictionary
import pandas as pd
from utils import load_octis_data, load_data, get_topics, compute_cv_coherence, calculate_npmi, compute_umass_coherence, compute_topic_diversity_coherence
corpus = ""


import sys
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    metric = sys.argv[2]
    beta_value = f"mom_{metric}"
    NUM_TOPICS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    NUM_ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    RUN_NUMBER = sys.argv[5] if len(sys.argv) > 5 else "oops"
    SEED = int(sys.argv[6]) if len(sys.argv) > 6 else "0"

else:
    dataset = "20NewsGroup"
    metric = "25_soc_pmi"
    beta_value = f"mom_{dataset.lower()}_{metric}"

path = "../mallet/bin/mallet"
dictionary, bow_corpus, doc_term_matrix, texts = load_octis_data(f"../data/{dataset}", to_save=False)
lda = LdaMallet(path, doc_term_matrix, random_seed=SEED, num_topics=NUM_TOPICS, optimize_interval=10, iterations=NUM_ITERATIONS, id2word=dictionary, beta_path=f"priors/{dataset}/{NUM_TOPICS}/{beta_value}.csv")
topics = get_topics(lda)

# print to a file
from contextlib import redirect_stdout
with open(f"{dataset.lower()}_svd.txt", "a", encoding="utf-8") as f:
    with redirect_stdout(f):
        print("\n###### Beta value: {} ###### \n ".format(beta_value))
        try:
            for topic_id in range(5):
                print(f"Topic {topic_id + 1}: \n{lda.show_topic(topic_id, topn=10)}")
        except Exception as e:
            print(f"Only 4 topics are available")
        compute_umass_coherence("Mallet", lda)
        compute_cv_coherence("Mallet", topics)
        compute_topic_diversity_coherence("Mallet", lda, 10)
        calculate_npmi(topics, texts, "Mallet")

# print to screen

print("\n###### Beta value: {} ###### \n ".format(beta_value))
try:
    for topic_id in range(5):
        print(f"Topic {topic_id + 1}: \n{lda.show_topic(topic_id, topn=10)}")
except Exception as e:
        print(f"Only 4 topics are available")

# compute_cv_coherence(metric, topics)
# compute_topic_diversity_coherence(metric, lda, 10)
# calculate_npmi(topics, texts, metric)

# save distribution to a file
def save_distributions(model, model_type, corpus):
    path = f"TopicDistributions/Distributions-Results/{RUN_NUMBER}/{dataset}/{NUM_TOPICS}"
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    # topic-word

    # Get topic-word matrix
    topic_word_matrix = model.get_topics()  # shape (num_topics, num_words)

    # Save to CSV
    topic_word_df = pd.DataFrame(topic_word_matrix, columns=[dictionary[i] for i in range(topic_word_matrix.shape[1])])
    topic_word_df.to_csv(f"{path}/{model_type}_topic_word_distribution.csv", index_label="Topic")


if RUN_NUMBER == "test":
    save_distributions(lda, "soc_mallet", corpus)
    cv = compute_cv_coherence(metric, topics)
    td = compute_topic_diversity_coherence(metric, lda, 10)
    npmi = calculate_npmi(topics, texts, metric)
    with open(f"Evaluations/results/{dataset}/PRISM.txt", "a") as f:
        f.write(f"Number of Topics: {NUM_TOPICS}\n")
        f.write(f"cv:{cv}\nTD:{td}\nnpmi:{npmi}\n\n")

if metric == "glove":
    save_distributions(lda, "glove_mallet", corpus)

if metric == "svd":
    save_distributions(lda, "svd_mallet", corpus)
