import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
books = pd.read_csv('Books.csv', dtype={'Year-Of-Publication': str})

users=pd.read_csv('Users.csv')
rating=pd.read_csv('Ratings.csv')
# print(books.head())
# print(users.head())
# print(rating.head())

# print(books.isnull().sum())
# print(users.isnull().sum())
# print(rating.isnull().sum())


# Top 10 publishers with the most books
top_publishers = books['Publisher'].value_counts().head(10)
sns.barplot(x=top_publishers.index, y=top_publishers.values)
plt.xticks(rotation=90)
plt.title("Top 10 Publishers with Most Books")
plt.xlabel("Publisher")
plt.ylabel("Number of Books")
# plt.show()


# Age Distribution4
plt.figure(figsize=(10,6))
sns.histplot(users['Age'], kde=True)
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Frequency')
# plt.show()

# Popularity Based Recommender System 

rating_with_name=rating.merge(books,on='ISBN')
num_rating_df=rating_with_name.groupby('Book-Title').count()['Book-Rating']
num_rating_df=num_rating_df.rename('num_ratings')

avg_rating_df=rating_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df=avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'})
# print(avg_rating_df)

popular_df=pd.merge(num_rating_df,avg_rating_df,on='Book-Title')

popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)
popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]
print(popular_df)

# Collaborative filtering 

x=rating_with_name.groupby('User-ID').count()['Book-Rating'] >200
padhe_likhe_user=x[x].index
# print(rating_with_name)
filter_rating=rating_with_name[rating_with_name['User-ID'].isin(padhe_likhe_user)]
# print('Filter')
# print(filter_rating)
y=filter_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index
final_rating=filter_rating[filter_rating['Book-Title'].isin(famous_books)]
# print(final_rating)

pt=final_rating.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)
# print(pt) 

from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(pt)
# print(similarity_score)

def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data

import pickle

# Assuming popular_df is your DataFrame
pickle.dump(popular_df, open('popular.pkl', 'wb'))
pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,open('similarity_scores.pkl','wb'))
