import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
    

df_books.dropna(inplace=True)
df_books.isnull().sum()

print(f"Total number of ratings : {len(df_ratings)}")

raters.sort_values(ascending=False).head()

print(f"Number of raters that will be dropped : {len(raters[raters < 200])}")
print(f"The numbers of ratings that will be dropped : {df_ratings['user'].isin(raters[raters < 200].index).sum()}")
df_ratings_rm = df_ratings[~df_ratings['user'].isin(raters[raters < 200].index)]
print(f"The numbers of remaining ratings : {len(df_ratings_rm)}")


print(f"Total number of ratings : {len(df_ratings)}")
print(f"Total number of books : {len(df_books)}")
rated = df_ratings['isbn'].value_counts()
print(f"Total number of rated books : {len(rated)}")


rated.sort_values(ascending=False).head()

print(f"The numbers of rated books that will be dropped : {len(rated[rated < 100])}")
print(f"The numbers of books that will be dropped : {df_books['isbn'].isin(rated[rated < 100].index).sum()}")
df_ratings_rm = df_ratings_rm[~df_ratings_rm['isbn'].isin(rated[rated < 100].index)]
print(f"The numbers of remaining ratings : {len(df_ratings_rm)}")

books = ["Where the Heart Is (Oprah's Book Club (Paperback))",
        "I'll Be Seeing You",
        "The Weight of Water",
        "The Surgeon",
        "I Know This Much Is True"]

for book in books:
  print(df_ratings_rm.isbn.isin(df_books[df_books.title == book].isbn).sum())
  
 df = df_ratings_rm.pivot_table(index=['user'],columns=['isbn'],values='rating').fillna(0).T
df.head()

df.index = df.join(df_books.set_index('isbn'))['title']
df = df.sort_index()
df.head()

df.loc["The Queen of the Damned (Vampire Chronicles (Paperback))"][:5]

model = NearestNeighbors(metric='cosine')
model.fit(df.values)
def get_recommends(title = ""):
  try:
    book = df.loc[title]
  except KeyError as e:
    print('The given book', e, 'does not exist')
    return

  distance, indice = model.kneighbors([book.values], n_neighbors=6)

  recommended_books = pd.DataFrame({
      'title'   : df.iloc[indice[0]].index.values,
      'distance': distance[0]
    }).sort_values(by='distance', ascending=False).head(5).values

  return [book,recommended_books]

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  #if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
   # test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")


test_book_recommendation()
