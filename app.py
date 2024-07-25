import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
data = pd.read_csv('recipe_recommendation_system_expanded.csv')

# Load the model and vectorizer
with open('recipe_recommender_model.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Vectorize Ingredients and Steps
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Ingredients'] + " " + data['Steps'])

# Function to get recommendations based on ingredients
def get_recommendations_by_ingredients(ingredients, cosine_sim=cosine_sim):
    ingredients_str = " ".join(ingredients)
    user_ingredients_vec = tfidf_vectorizer.transform([ingredients_str])
    cosine_sim_ingredients = linear_kernel(user_ingredients_vec, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim_ingredients[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recipe_indices = [i[0] for i in sim_scores[:10]]
    return data.iloc[recipe_indices]

# Function to filter recipes based on various criteria
def filter_recipes(ingredients=None, difficulty=None, cuisine=None):
    filtered_data = data.copy()
    
    if ingredients:
        ingredients_str = " ".join(ingredients)
        user_ingredients_vec = tfidf_vectorizer.transform([ingredients_str])
        cosine_sim_ingredients = linear_kernel(user_ingredients_vec, tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim_ingredients[0]))
        recipe_indices = [i[0] for i in sim_scores[:10]]
        filtered_data = filtered_data.iloc[recipe_indices]
    
    if difficulty:
        filtered_data = filtered_data[filtered_data['Difficulty'].str.contains(difficulty, case=False)]
    
    if cuisine:
        filtered_data = filtered_data[filtered_data['Cuisine'] == cuisine]
    
    return filtered_data

# Function to display recipe details
def display_recipe_details(recipe):
    st.write(f"**Ingredients:** {recipe['Ingredients']}")
    st.write(f"**Cuisine:** {recipe['Cuisine']}")
    st.write(f"**Steps:** {recipe['Steps']}")
    st.write(f"**RecipeID:** {recipe['RecipeID']}")
    st.write(f"**DietaryRestrictions:** {recipe['DietaryRestrictions']}")
    st.write(f"**PreparationTime:** {recipe['PreparationTime']} minutes")
    st.write(f"**CookingTime:** {recipe['CookingTime']} minutes")
    st.write(f"**Difficulty:** {recipe['Difficulty']}")
    st.write(f"**UserRating:** {recipe['UserRating']}")
    st.write(f"**Servings:** {recipe['Servings']}")

# Streamlit web application
st.title('Recipe Recommendation System')

# Filters for recommendation
st.header('Filter Recipes')

ingredients_input = st.text_area('Ingredients (separated by commas, optional)')
difficulty = st.selectbox('Difficulty (optional)', ['','Easy', 'Medium', 'Hard'])
cuisine_list = data['Cuisine'].unique().tolist()
cuisine = st.selectbox('Cuisine (optional)', [''] + cuisine_list)

if st.button('Show Recipes'):
    user_ingredients = [ingredient.strip() for ingredient in ingredients_input.split(',')] if ingredients_input else None
    filtered_recipes = filter_recipes(ingredients=user_ingredients, difficulty=difficulty, cuisine=cuisine)
    
    st.write(f"**Found {len(filtered_recipes)} Recipes:**")
    for i, recipe in filtered_recipes.iterrows():
        st.write(f"### {i + 1}. {recipe['RecipeName']}")
        display_recipe_details(recipe)

# Run the Streamlit app
# Save this script as `app.py` and run `streamlit run app.py` in the terminal
