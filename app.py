#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset

hotel_review_d = load_dataset('ashraq/hotel-reviews', split = 'train')


# In[ ]:


hotel_review_df = hotel_review_d.to_pandas()


# In[ ]:


from llama_index.core import Document
documents = [Document(text=row['review'], metadata={'hotel': row['hotel_name']}) for index, row in hotel_review_df.iterrows()]


# In[ ]:


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# In[ ]:


from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3:instruct", request_timeout=60.0)

response = llm.complete("What is the capital of France?")
print(response)


# In[ ]:


from llama_index.core import Settings

Settings.llm = llm
Settings.chunk_size = 512
Settings.embed_model = embed_model


# In[ ]:


from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(
    documents
)


# In[ ]:


index.storage_context.persist(persist_dir="hotel")


# In[ ]:


from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="hotel")

# load index
vector_index = load_index_from_storage(storage_context)


# In[ ]:


query_engine = vector_index.as_query_engine(similarity_top_k=10)


# In[ ]:


import pandas as pd
import geopy
from geopy.geocoders import Nominatim
import random

# assume your original dataframe is called df

# extract unique hotel names into a new dataframe
hotel_names_df = pd.DataFrame(hotel_review_df['hotel_name'].unique(), columns=['hotel_name'])

# create a geolocator object
geolocator = Nominatim(user_agent="my_app")

# create two new columns for longitude and latitude
hotel_names_df['longitude'] = None
hotel_names_df['latitude'] = None

# define a function to generate a random location in London
def random_london_location():
    # London's bounding box: 51.2868° N, 0.0053° W, 51.6913° N, 0.1743° E
    lat = random.uniform(51.2868, 51.6913)
    lon = random.uniform(-0.1743, 0.0053)
    return lat, lon

# loop through each hotel name and get its lat/long info
for index, row in hotel_names_df.iterrows():
    hotel_name = row['hotel_name'] + ', London'
    location = geolocator.geocode(hotel_name)
    if location:
        hotel_names_df.at[index, 'longitude'] = location.longitude
        hotel_names_df.at[index, 'latitude'] = location.latitude
    else:
        lat, lon = random_london_location()
        hotel_names_df.at[index, 'longitude'] = lon
        hotel_names_df.at[index, 'latitude'] = lat
        print(f"Could not find location for {hotel_name}, using random location instead")

# print the resulting dataframe
print(hotel_names_df)


# In[ ]:


import gradio as gr
import pandas as pd
import folium

# assume hotel_names_df is the dataframe with hotel names and lat/long info

def query(text):
    z = query_engine.query(text)
    return z

def generate_map(hotel_names):
    # generate map using Folium
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
    for hotel_name in hotel_names:
        lat = hotel_names_df[hotel_names_df['hotel_name'] == hotel_name]['latitude'].values[0]
        lon = hotel_names_df[hotel_names_df['hotel_name'] == hotel_name]['longitude'].values[0]
        folium.Marker([lat, lon], popup=hotel_name).add_to(m)
    return m._repr_html_()

def interface(text):
    z = query(text)
    response = z.response
    hotel_names = list(set([z.source_nodes[i].metadata['hotel'] for i in range(len(z.source_nodes))]))
    map_html = generate_map(hotel_names)
    return response, map_html



# In[ ]:


query_engine('Which hotel had the best food')


# In[103]:


import gradio as gr
import folium

with gr.Blocks(theme=gr.themes.Glass().set(block_title_text_color= "black", body_background_fill="black", input_background_fill= "black", body_text_color="white")) as demo:
    
    gr.Markdown("<style>h1 {text-align: center;display: block;}</style><h1>Hotel Reviews Chatbot</h1>")
    with gr.Row():
        output_text = gr.Textbox(lines=20)
        map_area = gr.HTML(value=folium.Map(location=[51.5074, -0.1278], zoom_start=12)._repr_html_())
        
    with gr.Row():
        input_text = gr.Textbox(label='Enter your query here')
        
    input_text.submit(fn=interface, inputs=input_text, outputs=[output_text, map_area])
                      
demo.launch(share=True)


# In[ ]:




