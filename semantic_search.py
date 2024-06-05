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

    #util.semantic_search：这个函数执行语义搜索，找到每个查询在语料库中得分最高的 top_k 个匹配项。 score_function=util.dot_score：使用点积作为相似度度量。
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=6)

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


#hit[k]['corpus_id']+2000*13 计算实际的 corpus_id，因为语料库嵌入是在 stored_embeddings 的后半部分。 输出每个查询匹配的语料库嵌入的 corpus_id。
# for i, hit in enumerate(hits):
#     # print("Query:", stored_sentences[i])
#     for k in range(len(hit)):
#         print(hit[k]['corpus_id']+2000*13, end=" ")
#         # print(hit[k]['score'])
#     print("")