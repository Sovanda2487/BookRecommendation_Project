import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="📚 Book Recommendation System", layout="wide", page_icon="📚")

# Load datasets
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')

# Cache resources to load them only once
@st.cache_resource
def load_models_and_data():
    with open('book_similarity_df.pkl', 'rb') as file:
        book_similarity_df = pickle.load(file)
    return book_similarity_df

book_similarity_df = load_models_and_data()

# Merge book details with ratings
merged_df = pd.merge(books[['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-M']], 
                     ratings, 
                     on='ISBN', 
                     how='inner')

# Top Rated Books (Popular)
def get_top_rated_books(num_books=151):
    top_books_isbn = ratings['ISBN'].value_counts().head(num_books).index
    top_books = books[books['ISBN'].isin(top_books_isbn)]
    return top_books[['Book-Title', 'Book-Author', 'Image-URL-M']].drop_duplicates()

# Collaborative Filtering Recommendation Function
def recommend_books_cf(book_title, num_recommendations=11):
    if book_title not in book_similarity_df.index:
        return f"Book '{book_title}' not found in the dataset."
    
    similar_books = book_similarity_df[book_title].sort_values(ascending=False)[1:num_recommendations + 1]
    recommendations = books[books['Book-Title'].isin(similar_books.index)].drop_duplicates('Book-Title')
    return recommendations[['Book-Title', 'Book-Author', 'Image-URL-M']]

# Streamlit App
st.title("📚 Book Recommendation System")
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {color: #1a5276; font-size: 2.5rem;}
    .sidebar .sidebar-content {background-color: #f4f4f4;}
</style>
""", unsafe_allow_html=True)

# Sidebar for User Options
st.sidebar.title("Recommendation Options")
option = st.sidebar.radio("Choose Recommendation Type", ["Collaborative Filtering", "Top 150 Most Rated"])

# Professional UI Layout for Top Rated Books
if option == "Top 150 Most Rated":
    st.subheader("📊 Top 150 Most Rated Books")
    popular_books_details = get_top_rated_books()
    
    # Display in a 3x50 grid (50 rows, 3 columns per row)
    for i in range(0, len(popular_books_details), 3):
        cols = st.columns(3, gap="medium")
        for idx, row in enumerate(popular_books_details.iloc[i:i+3].itertuples()):
            with cols[idx]:
                st.image(row._3, width=150, caption=row._1)
                st.caption(f"by {row._2}")

# Collaborative Filtering Recommendation
elif option == "Collaborative Filtering":
    st.subheader("🔍 Collaborative Filtering Recommendations")
    
    with st.form("recommend_form"):
        book_title = st.text_input("Enter a Book Title:")
        submitted = st.form_submit_button("Get Recommendations")
        
        if submitted:
            if book_title:
                recommendations = recommend_books_cf(book_title, 10)
                if isinstance(recommendations, str):
                    st.error(recommendations)
                else:
                    st.subheader("📚 Recommended Books:")
                    for _, row in recommendations.iterrows():
                        st.image(row['Image-URL-M'], width=150)
                        st.markdown(f"**{row['Book-Title']}** by {row['Book-Author']}")
                        st.markdown("<hr>", unsafe_allow_html=True)  # Divider between books
            else:
                st.warning("Please enter a book title.")

# Footer
st.markdown(
    "<hr style='border:1px solid #d3d3d3;'>",
    unsafe_allow_html=True
)
