from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


driver = webdriver.Chrome(executable_path='/Users/serg/Desktop/projects/youtube_parser/chromedriver')
driver.get("https://www.youtube.com/results?search_query=python")

user_data = driver.find_elements_by_xpath('//*[@id="thumbnail"]')
links = []
for i in user_data:
    links.append(i.get_attribute('href'))

df = pd.DataFrame(columns=['link', 'title', 'description', 'category'])

wait = WebDriverWait(driver, 10)

v_category = "Python"

for x in links:
    driver.get(x)
    v_id = x.strip('https://www.youtube.com/watch?v=')
    v_title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.title yt-formatted-string"))).text
    v_description = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#description yt-formatted-string"))).text
    df.loc[len(df)] = [v_id, v_title, v_description, v_category]

frames = [df_travel, df_science, df_food, df_manufacturing, df_history, df_artndance]
df_copy = pd.concat(frames, axis=0, join='outer', join_axes=None, ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False, copy=True)

df_link = pd.DataFrame(columns=["link"])
df_title = pd.DataFrame(columns=["title"])
df_description = pd.DataFrame(columns=["description"])
df_category = pd.DataFrame(columns=["category"])
df_link['link'] = df_copy['link']
df_title['title'] = df_copy['title']
df_description['description'] = df_copy['description']
df_category['category'] = df_copy['category']

nltk.download('stopwords')

corpus = []
for i in range(0, 8375):
    review = re.sub('[^a-zA-Z]', ' ', df_title['title'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
corpus1 = []
for i in range(0, 8375):
    review = re.sub('[^a-zA-Z]', ' ', df_description['description'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)

dftitle = pd.DataFrame({'title': corpus})
dfdescription = pd.DataFrame({'description': corpus1})
dfcategory = df_category.apply(LabelEncoder().fit_transform)
df_new = pd.concat([df_link, dftitle, dfdescription, dfcategory], axis=1, join_axes=[df_link.index])
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus, corpus1).toarray()
y = df_new.iloc[:, 3].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)
