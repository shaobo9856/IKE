from sentence_transformers import SentenceTransformer
import pickle
import json

model = SentenceTransformer('all-MiniLM-L6-v2')
with open('./data/zsre.json', 'r') as f:
    lines = json.load(f)

sentences = []
subjects = []

for line in lines:
    en_data = line['en']
    new_fact = en_data['new_fact']
    prompt = en_data['prompt']
    type_ = en_data['type']
    subject = en_data['id']  # 假设 id 是唯一标识符

    if type_ == 'copy':
        sentences.append(f"New Fact: {new_fact}\nPrompt: {prompt}")
    elif type_ == 'update':
        sentences.append(f"New Fact: {new_fact}\nPrompt: {prompt}")
    elif type_ == 'retain':
        sentences.append(f"New Fact: {new_fact}\nPrompt: {prompt}")

    subjects.append(subject)



embeddings = model.encode(sentences, show_progress_bar=True)

#Store sentences & embeddings on disc
with open('embeddings.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'subjects': subjects}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
# with open('embeddings.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']