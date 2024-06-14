from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import json

with open('embeddings.pkl', "rb") as fIn:
    data = pickle.load(fIn)
    query_ids = data['query_ids']
    corpus_ids = data['corpus_ids']

    query_sentences = data['query_sentences']
    query_embeddings = torch.tensor(data['query_embeddings']).to('cuda')
    query_embeddings = util.normalize_embeddings(query_embeddings)

    corpus_sentences = data['corpus_sentences']
    corpus_embeddings = torch.tensor(data['corpus_embeddings']).to('cuda')
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=3)

    # print(hits)

    ids = []
    for i, hit in enumerate(hits):
        query_id = query_ids[i]
        retrieved_ids = []
        for k in range(len(hit)):
            retrieved_ids.append(corpus_ids[hit[k]['corpus_id']])
        
        ids.append({'query_id': query_id,
                'corpus_ids': retrieved_ids})

with open('corpus_idx.json', 'wt') as f_out:
    json.dump(ids, f_out, ensure_ascii=False, indent=2)

