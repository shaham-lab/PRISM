import graph_utils as gu
import numpy as np

def main(n_comp=5, metrics="pmi", k=10, use_soc=False, use_glove=False, is_first=False):
    path_data = f"../data/{dataset}/{dataset}.json"
    corpus, dictionary = gu.load_data(path_data, type="json")
    is_first_pmi = is_first
    origin_metric_name = metrics 
    
    if is_first_pmi:
        pmi_matrix, vocab = gu.compute_ppmi_across_window_sizes(corpus=corpus, dataset=dataset, kind=metrics, is_first=is_first)
            is_first_pmi = False
    else:
        path_load = f"data/{dataset}"
        pmi_matrix, vocab = gu.load_pmi_matrix(path=path_load, window_size=w)
        
    if use_soc:
        print("soc matrix applied")
        affinity_matrix = gu.get_cosine_similarity(pmi_matrix) # second order pmi
        metrics = f"{k}_soc_" + origin_metric_name
    elif use_glove:
        print("glove matrix applied")
        metrics = "glove"
        affinity_matrix = gu.get_glove_matrix(dictionary, glove_path="glove/glove.6B.100d.txt", path_save="")
    
    else:
        metrics = f"{k}_" + metrics
        affinity_matrix = pmi_matrix
        

    gu.create_prior(affinity_matrix, corpus, vocab, dictionary, dataset=dataset, w=None,
                        n_components=n_comp, k=k,
                        path_to_save=path_to_save, 
                        metric=metrics, 
                        is_bayesian=True, 
                        smooth=True,
                        eval_prior=True,
                        is_first=is_first
                        )
      
    
if __name__ == "__main__":
    import sys
    import random
    import numpy as np
    import torch
    
    argv = sys.argv

    argc = len(argv)    
    if argc > 1:
        n_comp = int(argv[1])
        dataset = argv[2]
        path_to_save = "priors/" + dataset
        metrics = argv[3]
        k = int(argv[4])
        is_first = bool(int(argv[5]))
        if len(argv) > 6:
            use_soc = 1
            use_glove = 0
        else:
            use_soc = 1
            use_glove = 0

    else:
        n_comp = 50
        dataset = "20NewsGroup"
        path_to_save = f"priors/{dataset}"
        metrics = "svd"
        use_glove = 0
        use_soc = 1
        k = 90
        is_first = False

    main(n_comp=n_comp, metrics=metrics, use_soc=use_soc, use_glove=False, k=k, is_first=is_first)

