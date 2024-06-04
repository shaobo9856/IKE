from sentence_transformers import SentenceTransformer, util
import torch
import pickle

with open('embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']

corpus_embeddings = torch.tensor(stored_embeddings[2000*13:]) # 包含第 2000*13 之后的嵌入，作为语料库。 代码中2000*13和13*2000是因为每个实例包含13个不同的句子嵌入。
query_embeddings = torch.tensor(stored_embeddings[:13*2000]) # 包含前 2000*13 的嵌入，作为查询。

corpus_embeddings = corpus_embeddings.to('cuda')
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to('cuda')
query_embeddings = util.normalize_embeddings(query_embeddings)

#util.semantic_search：这个函数执行语义搜索，找到每个查询在语料库中得分最高的 top_k 个匹配项。 score_function=util.dot_score：使用点积作为相似度度量。
hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=64)

# print(hits)

#hit[k]['corpus_id']+2000*13 计算实际的 corpus_id，因为语料库嵌入是在 stored_embeddings 的后半部分。 输出每个查询匹配的语料库嵌入的 corpus_id。
for i, hit in enumerate(hits):
    # print("Query:", stored_sentences[i])
    for k in range(len(hit)):
        print(hit[k]['corpus_id']+2000*13, end=" ")
        # print(hit[k]['score'])
    print("")