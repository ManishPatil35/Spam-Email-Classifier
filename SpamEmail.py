
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer   # For converting text to matrix of token counts
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
nltk.download('stopwords')

# Load your CSV file from the download folder
# Replace 'emails.csv' with the actual name of your CSV file
csv_file_path = r"C:\Users\MANISH\Desktop\Spam_Classifier_Python\email_spam.csv"
df = pd.read_csv(csv_file_path)

df.drop_duplicates(inplace=True) #Removing the missing Values

def process_text(text):

  #Remove Punctuation
  nopunc =[char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  #Remove Stopwords
  clean_word = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

  return clean_word

# Convert a collection of text into matrix of tokens
message_bow = CountVectorizer(analyzer = process_text).fit_transform(df['text'])

#Split data into 80% Training and 20% Testing
X_train , X_test , y_train , y_test = train_test_split(message_bow , df['type'] , test_size = 0.20 , random_state=0)

#Create and train naive bayes classifier
classifier = MultinomialNB().fit(X_train,y_train)

#Evaluate Model on Test Dataset
pred_test = classifier.predict(X_test)

accuracy_t = accuracy_score(y_test, pred_test)
conf_matrix_t = confusion_matrix(y_test, pred_test)
clasf_report_t = classification_report(y_test,pred_test)

print(f"Accuracy: {accuracy_t}")
print()
print(f"Confusion Matrix:\n{conf_matrix_t}")
print()
print(f"Classification Report:\n{clasf_report_t}")


