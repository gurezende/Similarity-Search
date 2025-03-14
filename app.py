import streamlit as st
import chromadb
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch
# bug fix to eliminate error message
torch.classes.__path__ = []

# Clear the system cache if this is the first run
chromadb.api.client.SharedSystemClient.clear_system_cache()

#--------------------------- Helper Functions ----------------------------

# Function to reduce dimensions of the embeddings
def reduce_dimensions(vectors):
    """
    Reduces the dimensionality of a given set of vectors from 384 to 3.
    
    Parameters
    ----------
    vectors : numpy.ndarray
        The vectors to reduce, of shape (n_samples, 384)
    
    Returns
    -------
    reduced_vectors : numpy.ndarray
        The reduced vectors, of shape (n_samples, 3)
    """
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    return reduced_vectors

#--------------------------- Streamlit App ----------------------------

# Set up the Streamlit app main page
st.set_page_config(page_title="Similarity Search Visualization",
                   page_icon=":sparkles:",
                   layout="wide")

# Page Title
st.title("Similarity Search Visualization")
st.write('Welcome to the Similarity Search Visualization App! This app allows you to perform similarity search on a vector database and visualize the results in a 3D plot.')

# Counter
counter = 0

# Sidebar for the Prepare Data and Clear Cache buttons
with st.sidebar:

    #Prepare Data
    st.header(":point_down: :red[Start Here]")
    st.subheader(":one: Click the button before running a search")
    prepare_data = st.button("Prepare the Data")
    counter += 1

    st.divider()

    # Clear Cache
    st.caption("Clear the System Cache and restart if you are finding issues with the app.")
    clear_cache = st.button("Clear Cache", on_click=lambda: chromadb.api.client.SharedSystemClient.clear_system_cache())


# ----- Query -------
# Text input from the user to query the vector database
query = st.text_input("Search Query", placeholder="Ask me something about Health (e.g. 'Quick Workout')", key="search_query")

# Search button. If pressed, query the vector database
viz = st.button("Search and Visualize Similarity", key="search_button")


# Condition to check if the Prepare Data button is pressed
if prepare_data:

    # ----- Print Log -------

    # Sidebar header
    st.header(":one: Step 1: Load and Prepare the Data")
    st.caption("There is a sample of Health News data pre-configured.")
    data = pd.read_json("healthnews.json", lines=True)
    st.session_state['df'] = data.sample(500)
    st.write(":white_check_mark: Data loaded successfully!")
    st.write(st.session_state['df'].sample(3))

    st.divider()

    # Client connection to the database
    st.header(":two: Step 2: Create Client Connection")
    st.caption("You need to connect to your vector database.")
    st.session_state['chroma_client'] = chromadb.PersistentClient()
    st.write(":white_check_mark: Client connected successfully!")

    st.divider()
    
    # Creating a collection
    st.header(":three: Step 3: Create a Collection and add the data")
    st.caption("A collection is like a table in a database.")
    collection = st.session_state['chroma_client'].get_or_create_collection("my_collection")
        
    # Adding data to the collection
    df = st.session_state['df']
    collection.add(
        documents=df["News"].tolist(), 
        metadatas=[{"Date": str(dt)} for dt in df["Date"].tolist()],
        ids=[str(i) for i in range(len(df))]
    ) 
    # renaming the session state collection
    st.session_state['collection'] = collection

    # Success message
    st.write(":white_check_mark: Collection created successfully!")
    st.success(":white_check_mark: Data Loaded!")


    # Showing data transformed to embeddings on screen as an array
    st.write("Data Transformed to Embeddings Vectors (partial view)")
    c = collection.get(ids=["0"], include=['embeddings'])
    st.caption(str(c['embeddings'][0][:20].tolist()))


# ----- Visualization -------
if viz: #if the search button is pressed

    # If this button is pressed before the Prepare Data button, show an error message
    if counter == 0:
        st.error("Please prepare the data before searching.")

    # Query the collection
    results = st.session_state['collection'].query(
        query_texts=query, # Chroma will embed this for you
        n_results=3 # how many results to return
    )

    # write the results
    st.write(
        pd.DataFrame({
            "Distance": [score for score in results['distances']][0],
            "News": [News for News in results['documents']][0]
        }) )

    # Prepare the data for plotting
    query_vector = SentenceTransformer('all-MiniLM-L6-v2').encode(query)
    result_vectors = SentenceTransformer('all-MiniLM-L6-v2').encode(results['documents'][0])

    # Combine vectors
    all_vectors = np.vstack([query_vector, result_vectors])

    # Reduce dimensions with PCA
    reduced_vectors = reduce_dimensions(all_vectors)

    # Create Plotly figure
    fig = go.Figure(data=[
        go.Scatter3d(
            x=[reduced_vectors[0, 0]],
            y=[reduced_vectors[0, 1]],
            z=[reduced_vectors[0, 2]],
            mode='markers',
            marker=dict(size=10, color='red'),  # Query point (red)
            name='Query'
        ),
        go.Scatter3d(
            x=reduced_vectors[1:, 0],
            y=reduced_vectors[1:, 1],
            z=reduced_vectors[1:, 2],
            mode='markers',
            marker=dict(size=6, color='blue'),  # Result points (blue)
            name='Results'
        )
    ])

    # Modify axis ranges
    fig.update_layout(scene = dict(
        xaxis = dict(range=[-.8, .8]),
        yaxis = dict(range=[-.8, .8]),
        zaxis = dict(range=[-.8, .8])
    ))

    # Show the plot
    st.plotly_chart(fig)