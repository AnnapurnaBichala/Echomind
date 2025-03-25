import streamlit as st
import pandas as pd

# Use caching to avoid reloading CSV every time
@st.cache_data
def load_data():
    return pd.read_csv('metadata.csv')

# Function to display the recommendation based on selected category and product
def recommend_product(selected_category, selected_product, metadata):
    # Filter metadata based on selected category
    filtered_data = metadata[metadata['main_category'] == selected_category]
    
    # Filter the product from the selected category
    product_data = filtered_data[filtered_data['parent_asin'] == selected_product]
    
    if not product_data.empty:
        # Display the relevant information if the product is found
        st.markdown(f"### **{product_data['title'].values[0]}**")
        
        # Handling price if it's NaN
        price = product_data['price'].values[0]
        if pd.isna(price):
            price = "Price not found"
        
        # Display product information
        st.write(f"**Average Rating:** {product_data['average_rating'].values[0]}")
        st.write(f"**Number of Ratings:** {product_data['rating_number'].values[0]}")
        st.write(f"**Price:** {price}")
    else:
        # If no results found
        st.warning("No recommendation found.")

# Streamlit UI with design and branding for Wells Fargo
st.set_page_config(page_title="Echo Mind", page_icon="🧠", layout="wide")

# Load data once and cache it
metadata = load_data()

# Custom Styling with Wells Fargo colors
st.markdown("""
    <style>
        .main {
            background-color: #f8f5f1;  /* Soft background color */
            padding: 40px;
            border-radius: 10px;
        }
        h1 {
            color: #9E1B32;  /* Wells Fargo Red */
        }
        .stButton>button {
            background-color: #9E1B32;  /* Wells Fargo Red */
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #6D1321;  /* Darker Wells Fargo Red */
        }
        .stSelectbox>div>div>input {
            border-radius: 8px;
        }
        .stSelectbox {
            margin-top: 10px;
            margin-bottom: 20px;
        }
        .stText {
            font-size: 18px;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
        .stSelectbox, .stTextInput {
            background-color: white;
            border: 2px solid #9E1B32; /* Red border for inputs */
        }
    </style>
""", unsafe_allow_html=True)

st.title("Welcome to Echo Mind 💡")
st.markdown("""
    **Echo Mind**: A smart product recommendation system designed for **Banking**. Select a product and get personalized details.
    This system provides tailored recommendations based on your selection.
""")

# Create a container with custom padding
with st.container():
    # Dropdown for selecting category
    category = st.selectbox('### Select Product Category:', metadata['main_category'].unique())

    # Filter products in the selected category only once, not on each interaction
    products_in_category = metadata[metadata['main_category'] == category]
    product_options = dict(zip(products_in_category['parent_asin'], products_in_category['title']))
    
    # Dropdown for selecting product
    product = st.selectbox('### Select a Product:', product_options.keys(), format_func=lambda x: product_options[x])

    # Recommendation button
    if st.button('Recommend Now', key="recommend_button"):
        recommend_product(category, product, metadata)
