import warnings

warnings.filterwarnings('ignore')
import email
import imaplib
import os
import re
import smtplib
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import HungarianStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('universal_tagset')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('treebank')
stop_words = set(stopwords.words('hungarian'))

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sklearn.feature_extraction.text import TfidfVectorizer

import openai
from nltk.tokenize import sent_tokenize

openai.organization =  os.getenv("OPEN_API_ORGANISATION") 
openai.api_key = os.getenv("OPEN_API_KEY") 
IMAP_HOST = os.getenv("IMAP_HOST")  
IMAP_USER = os.getenv("IMAP_USER") 
IMAP_PASS = os.getenv("IMAP_PASS")  

# SMTP
SMTP_HOST = os.getenv("SMTP_HOST") 
SMTP_PORT = int(os.getenv("SMTP_PORT") )
SMTP_USER = os.getenv("SMTP_USER") 
SMTP_PASS = os.getenv("SMTP_PASS")  

# 
KNOWLEDGE_BASE=os.getenv("KNOWLEDGE_BASE") 

# MODE -> CHAT/MAILHANDLER
MODE = os.getenv("MODE")
MAX_HISTORY = os.getenv("MAX_HISTORY")
MAX_QUESTION_TO_RESPONS = int(os.getenv("MAX_QUESTION_TO_RESPONS"))
MAX_NUMBER_OF_RELEVANT_SENTENCES = int(os.getenv("MAX_NUMBER_OF_RELEVANT_SENTENCES"))
MODEL = "gpt-3.5-turbo"

#
# Extract question
# This part get all the questions from the Hungarian input text.
# Can be used for example with ChatGPT in the prompt creation
#

# Define the question words in Hungarian
question_words = ["hány","hányadik","hányadikor","hányféle","hányig","hánykor","hánytól","hogyan","hol","honnan",
                "honnét","hova","hová","ki","kicsoda","kié","kiért","kihez","kik","kiként","kiket","kim","kinek","kinél","kitől",
                "kivel","meddig","melyikőtök","melyiktek","mekkora","mely","melyek","melyik","melyikből","melyikért","melyiknél",
                "mennyi","mennyire","merre","merről","mettől","mi","miben","micsoda","miért","miféle","mihez","mik","miként","miképp",
                "miképpen","mikért","miket","mikor","mikortól","milyen","milyenek","milyenekért","milyenkor","min","minek","minél",
                "mióta","mire","miről","mit","mitől","mivel"]

urgent_words = ['halaszthatatlan', 'elodázhatatlan', 'elnapolhatatlan', 'mielőbbi', 'azonnali', 'sürgős', 
                'fontos', 'nélkülözhetetlen', 'alapvető', 'lényeges', 'elemi', 'lényegbeli', 'központi', 'középponti', 
                'esszenciális', 'elengedhetetlen', 'jelentős', 'releváns', 'jelentőségteljes', 'meghatározó', 'létfontosságú',
                'kulcsfontosságú', 'kardinális', 'hangsúlyos', 'kulcsponti', 'sarkalatos', 'lényegi', 'súlyponti', 'mérvadó', 
                'pótolhatatlan', 'fő', 'jelentékeny' ]
#
# Order the questions by importance (defined in the questions), and complexity
#
def _question_priority(question : str): 
    words = question.split()
    # Check for urgent or important keywords
    for i in range(len(words)):
        word = words[i].lower()
        if word in urgent_words:
            return 1
    # Check for length and complexity (has , to separate multiple part of the question)
    if len(question) > 100 or "," in question:
        return 3
    if len(question) > 50 :
        return 2
    return 4

# Filter out non-question sentences
def _filter(txt : str):
    # Split the email into sentences
    sentences = re.split('[.?!]', txt)
    # Loop through each sentence and find the question words
    questions = []
    for sentence in sentences:
        # clean the sentence
        sentence = sentence.strip().replace("\n", "")
        sentence = re.sub(' +', ' ', sentence)
        if sentence.endswith('?'):
            questions.append(sentence)
            continue
        words = sentence.split()
        for i in range(len(words)):
            word = words[i].lower()
            if word in question_words:
                if sentence.endswith('?'):
                    questions.append(sentence)
                else:
                    questions.append(sentence + '?')
                break
    return questions

def get_questions(text :str, top_k : int):
    return sorted(_filter(text), key=_question_priority)[:top_k]

def get_content():
    # Load the knowledge base from a pdf
    all_pages_text = []
    with open(KNOWLEDGE_BASE, 'rb') as file:
        for page_layout in extract_pages(file):
            page_text = ""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text += element.get_text().strip() + " "
            all_pages_text.append(page_text)
    all_pages_text = " ".join(all_pages_text)
    # Clean the text
    all_pages_text = re.sub(r'\n+', ' ', all_pages_text)
    all_pages_text = re.sub(r'\t+', ' ', all_pages_text)
    all_pages_text = re.sub(r'\s+', ' ', all_pages_text)

    # Get all the sentences
    sentences = sent_tokenize(all_pages_text)
 
    return sentences



def get_relevant(question, sentences):
    # Similarity search
    # Get the relevant sentences based on list of question
    # stemmer_ss = SnowballStemmer("hungarian")   
    # stemmed_sentences = []
    # for sentence in sentences:
    #     stemmed_sentence = [stemmer_ss.stem(word) for word in sentence]
    #     stemmed_sentences.append(stemmed_sentence)  

    # print(stemmed_sentences[:10])
    # Vectorize sentences
    tfidf = TfidfVectorizer(tokenizer=word_tokenize, stop_words=list(stop_words))
    tfidf.fit(sentences)
    # Vectorise questions
    question_vector = tfidf.transform(question)
    # Vectorise sentences and get the similarities
    sim_scores = tfidf.transform(sentences).dot(question_vector.T).toarray().ravel()
    #print('sim_score:',sim_scores)
    # select the top  relevant sentences
    top_sentences = []
    for i in sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True):
        if len(top_sentences) < MAX_NUMBER_OF_RELEVANT_SENTENCES and sim_scores[i] > 0.1:
            top_sentences.append(sentences[i])
    #print('top:',top_sentences)
    return top_sentences




def get_response( head, questions,sentences, history):
    chatbot_response = []
    relevant_sentences = get_relevant(questions,sentences)
    for question in questions :
        while True :
            try :
                message = head
                message[0]['content'] = message[0]['content'].replace('<knowledge_base>'," ".join(relevant_sentences))
                # message.extend(history[-3:])
                message.append({"role": "user", "content": f"Ha szerepel a tudásbázisban akkor válaszolj: {question}, de csak a Tudásbázis alapján válaszolhatsz!Mást ne is mondj!"})
                # Get the response from GPT-3
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=message,
                    max_tokens=2048, 
                    stop=None,
                    temperature=0,
                    n = 1 # how many anwser to create?
                )
                break
            except Exception as e:
                print (type(e), e)
                print("Clear history")
                history = []
        #print(response)
        # Extract the response from the response object
        response_text = response['choices'][0].message['content'].strip()
        chatbot_response.append({"role": "assistant", "content": f"{response_text}"})
        # Add response as a history element
        history.append({"role": "user", "content": f"{question}, de csak a Tudásbázis alapján válaszolhatsz!"})
        history.append({"role": "assistant", "content": f"{response_text}"})
    return chatbot_response

#
# CHAT mode
#
def chat_mode():
    history = []
    head = [{"role": "system", 
            "content": "Legyél chatbot assistant \n\nTudásbázis:\n\n<knowledge_base>\n\n"},
        {"role": "user", "content": "A felhasználó neve áll a családi névből és az utónévből, egymás után írva"},
      ]
    sentences = get_content()
    print('Sentence num:',len(sentences))

    while True:
        #print("history:", history)
        user_input = input("User: ")
        if user_input == "exit":
            break
        chatbot_response = get_response( head, [user_input],sentences, history)
        print(f"Chatbot:{chatbot_response[0]['content']}\n-----------------")



#
# MAIL_MODE
#
def _get_email_text(msg):
    # A levél tartalmának feldolgozása
    # Az alábbi példa csak visszaadja a teljes szöveget
    return msg.get_payload()

def process_request(sentences,questions):

    # Select relevat content
    relevant_sentences = []
    for question in questions:
        for sentence in sentences:
            # Ha a mondatban szerepel a kulcsszó, akkor hozzáadjuk a releváns mondatok listájához
            if re.search(rf'\b{re.escape(question)}\b', sentence):
                relevant_sentences.append(sentence)

        # Join selected sentences to text
        knowledge_base = ' '.join(relevant_sentences[:10]) # max the first 10
        message = [{"role": "system", 
                    "content": f"Legyél chatbot assistant \n\nTudásbázis:\n\n{knowledge_base}\n\n"},
                ]

        # ChatGPT-nél kérdezünk a releváns szövegrészre
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=message,
            max_tokens=2048, 
            stop=None,
            temperature=0,
            n = 1 # how many anwser to create?
        )
        return response

def read_unseen_mails(head, sentences, history):
    # BConnect to thw mail server
    imap_host = IMAP_HOST
    imap_user = IMAP_USER
    imap_pass = IMAP_PASS

    imap = imaplib.IMAP4_SSL(imap_host)
    imap.login(imap_user, imap_pass)
    imap.select('INBOX')

    # Download unseen mails
    status, messages = imap.search(None, 'UNSEEN')
    messages = messages[0].split(b' ')

    for mail in messages:
        # Adoenload one mail
        res, msg = imap.fetch(mail, '(RFC822)')

        for response in msg:
            if isinstance(response, tuple):
                # get the content and the headers
                msg = email.message_from_bytes(response[1])

                # Addresses
                sender = msg['From']
                recipient = msg['To']

                # Process the content
                text = _get_email_text(msg)
                questions= get_questions(text , top_k = MAX_QUESTION_TO_RESPONS)

                response=  get_response( head, questions,sentences, history)
                # Válasz e-mail összeállítása
                reply = email.message.EmailMessage()
                reply['To'] = sender
                reply['Subject'] = f"RE: {msg['Subject']}"
                reply.set_content(f"Válaszom a levelére:{response} \n---------------------------\n{text}\n\n")

                # SMTP szerveren keresztül elküldi a választ
                smtp_host = SMTP_HOST
                smtp_port = SMTP_PORT
                smtp_user = SMTP_USER
                smtp_pass = SMTP_PASS

                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(reply)
                    print(f"Válasz elküldve a következő címre: {sender}")


def email_handler_mode():
    history = []
    head = [{"role": "system", 
            "content": "Legyél chatbot assistant \n\Tudásbázis:\n\n<knowledge_base>\n\n"},
        {"role": "user", "content": "A felhasználó neve áll a családi névből és az utónévből, egymás után írva"},
      ]
    sentences = get_content()
    while True:
       
       read_unseen_mails(head, sentences, history)
       time.sleep(60)
 

def test() :
   sentences = get_content()
   print('Sentences:',sentences)
   relevant = get_relevant(['Mi a NN-Teladoc'],sentences)
   print('Relevants:', relevant)



if __name__ == "__main__":
 
    # test()
    # sys.exit()
    if MODE == "CHAT" :
        chat_mode()
    else :
        email_handler_mode()