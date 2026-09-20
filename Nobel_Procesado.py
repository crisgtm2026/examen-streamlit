import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

st.write(''' # Predicción de categoría de Premio Nobel ''')
st.image("Nobel.png", caption="Su creador fue el inventor sueco Alfred Nobel mediante su testamento en 1895.")

st.header('Texto')

def user_input_features():
    # Entrada
    texto = st.text_area("Introduce la motivación o aportación científica/humanística a evaluar", height=150)
    
    user_input_data = {'Text': texto}
    
    features = pd.DataFrame(user_input_data, index=[0])
    
    return features

df = user_input_features()

nobel = pd.read_csv('tabla_concatenada.csv')

nobel['Label'] = nobel['Category'].str.lower().map(
    {'physics': 0, 'medicine': 1, 'peace': 2, 'literature': 3, 'chemistry': 4, 'economics': 5}
)

X = nobel['clean_motivation']
y = nobel['Label']

vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_dtm)

nb = MultinomialNB()
nb.fit(X_tfidf, y)

st.subheader('Predicción')

if st.button('Clasificar'):
    if df['Text'].iloc[0].strip() == "":
        st.warning("Por favor, ingresa un texto para evaluar.")
    else:
        df_dtm = vect.transform(df['Text'])
        df_tfidf = tfidf_transformer.transform(df_dtm)
        prediction = nb.predict(df_tfidf)[0]
#{'physics':0, 'medicine':1, 'peace':2, 'literature':3, 'chemistry':4, 'economics':5}
#'Physics', 'Medicine', 'Peace', 'Literature', 'Chemistry', 'Economics'
        if prediction == 0:
            st.write('Physics')
        elif prediction == 1:
            st.write('Medicine')
        elif prediction == 2:
            st.write('Peace')
        elif prediction == 3:
            st.write('Literature')
        elif prediction == 4:
            st.write('Chemistry')
        elif prediction == 5:
            st.write('Economics')
        else:
            st.write('Sin predicción')
