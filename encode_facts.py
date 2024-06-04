from sentence_transformers import SentenceTransformer
import pickle
import json
import sys

model = SentenceTransformer('all-MiniLM-L6-v2')
with open('./data/zsre.json', 'r') as f:
    lines = json.load(f)

sentences = []
query_ids = []
corpus_ids = []
query_sentences = []
corpus_sentences = []

for line in lines:
    en_data = line['en']
    new_fact = en_data['new_fact']
    prompt = en_data['prompt']
    type_ = en_data['type']
    id = en_data['id']

    if id < 10:
        query_sentences.append(f"New Fact: {new_fact}\nPrompt: {prompt}")
        query_ids.append(id)
    else:
        corpus_sentences.append(f"New Fact: {new_fact}\nPrompt: {prompt}")
        corpus_ids.append(id)


query_embeddings = model.encode(query_sentences, show_progress_bar=True)
corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True)


#Store sentences & embeddings on disc
with open('embeddings.pkl', "wb") as fOut:
    pickle.dump({'query_sentences': query_sentences, 'query_embeddings': query_embeddings, 'query_ids': query_ids,
                'corpus_sentences': corpus_sentences, 'corpus_embeddings': corpus_embeddings, 'corpus_ids': corpus_ids,}, 
                fOut, protocol=pickle.HIGHEST_PROTOCOL)
