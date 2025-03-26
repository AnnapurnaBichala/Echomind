import streamlit as st
import pandas as pd
import json

from recommend import recommend_from_items, item_id2num, load_model
from personalize import generate_personalized_reason

st.set_page_config(page_title="Echo Mind", page_icon="🧠", layout="wide")


weights_path = "checkpoints/model_weights.pt"
metadata_path = "../data/amazon-electronics/filtered_item_metadata.csv"


model = load_model(weights_path)

@st.cache_data
def load_data():
    return pd.read_csv(metadata_path)

metadata = load_data()


st.markdown("""
    <style>
        .main { background-color: #f8f5f1; padding: 40px; border-radius: 10px; }
        h1 { color: #9E1B32; }
        .stButton>button {
            background-color: #9E1B32; color: white; font-size: 16px;
            border-radius: 8px; padding: 10px 20px; border: none;
        }
        .stButton>button:hover { background-color: #6D1321; }
        .stSelectbox, .stTextInput {
            background-color: white; border: 2px solid #9E1B32;
            border-radius: 8px; padding: 6px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Welcome to Echo Mind 💡")
st.markdown("""
    **Echo Mind**: A smart product recommendation system.  
    Select **two products** to get personalized, hybrid-based recommendations.
""")

category = st.selectbox('### Select Product Category:', metadata['main_category'].unique())
products_in_category = metadata[metadata['main_category'] == category]
product_options = dict(zip(products_in_category['parent_asin'], products_in_category['title']))

product1 = st.selectbox('### Select First Product:', product_options.keys(), format_func=lambda x: product_options[x])
product2 = st.selectbox('### Select Second Product:', product_options.keys(), format_func=lambda x: product_options[x], index=1 if len(product_options) > 1 else 0)

if st.button('Recommend Now', key="recommend_button"):
    st.subheader("Selected Products")
    for asin in [product1, product2]:
        prod = products_in_category[products_in_category['parent_asin'] == asin]
        if not prod.empty:
            st.markdown(f"**{prod['title'].values[0]}**")
            st.write(f"**Average Rating:** {prod['average_rating'].values[0]}")
            st.write(f"**Price:** {prod['price'].values[0] if not pd.isna(prod['price'].values[0]) else 'N/A'}")
        else:
            st.warning(f"Product {asin} not found in metadata.")

    item1_id = item_id2num[product1]
    item2_id = item_id2num[product2]

    recommended_ids = recommend_from_items(model, item1_id, item2_id)

    liked_titles = [
        product_options[product1],
        product_options[product2]
    ]

    st.markdown("### 🧠 Recommended Products")
    st.markdown("_Here are some items you might like based on your preferences._")
    for rec_asin in recommended_ids:
        if rec_asin:
            rec_row = metadata[metadata['parent_asin'] == rec_asin]

            title = rec_row['title'].values[0] if not rec_row.empty else "Unknown Title"
            rating = rec_row['average_rating'].values[0] if not rec_row.empty else "N/A"
            price = rec_row['price'].values[0] if not rec_row.empty and not pd.isna(rec_row['price'].values[0]) else "N/A"

            st.markdown(f"**{title}**")
            st.write(f"ASIN: `{rec_asin}`")
            st.write(f"Rating: {rating}")
            st.write(f"Price: {price}")

            try:
                explanation = generate_personalized_reason(liked_titles, title)
                st.markdown(f"> 💬 *{explanation}*")
            except Exception as e:
                st.warning("Could not generate personalized explanation.")
                print("OpenAI API error:", e)
                
            st.markdown("---")