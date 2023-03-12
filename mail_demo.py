import warnings

warnings.filterwarnings('ignore')

import email
import imaplib
import logging
import os
import re
import smtplib
import time
from datetime import datetime
from email.header import decode_header
from typing import List

from dotenv import load_dotenv

load_dotenv()

import nltk
from nltk.stem.snowball import SnowballStemmer
#from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stemmer = SnowballStemmer("hungarian")
STOPWORDS = set(nltk.corpus.stopwords.words('hungarian'))

import urllib.parse

import openai
# webscraping part
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
MAX_URL_IN_WEBSEARCH = int(os.getenv("MAX_URL_IN_WEBSEARCH"))
MAX_QUESTION_TO_RESPOND = int(os.getenv("MAX_QUESTION_TO_RESPOND"))
MAX_NUMBER_OF_RELEVANT_SENTENCES = int(os.getenv("MAX_NUMBER_OF_RELEVANT_SENTENCES"))

MODEL = "gpt-3.5-turbo"


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
# Extract questions
# This part get all the questions from the Hungarian input text.
# Can be used for example with ChatGPT in the prompt creation
#

# Define the question words in Hungarian
QUESTION_WORDS = ["hány","hányadik","hányadikor","hányféle","hányig","hánykor","hánytól","hogyan","hol","honnan",
                "honnét","hova","hová","ki","kicsoda","kié","kiért","kihez","kik","kiként","kiket","kim","kinek","kinél","kitől",
                "kivel","meddig","melyikőtök","melyiktek","mekkora","mely","melyek","melyik","melyikből","melyikért","melyiknél",
                "mennyi","mennyire","merre","merről","mettől","mi","miben","micsoda","miért","miféle","mihez","mik","miként","miképp",
                "miképpen","mikért","miket","mikor","mikortól","milyen","milyenek","milyenekért","milyenkor","min","minek","minél",
                "mióta","mire","miről","mit","mitől","mivel"]

URGENT_WORDS = ['halaszthatatlan', 'elodázhatatlan', 'elnapolhatatlan', 'mielőbbi', 'azonnali', 'sürgős', 
                'fontos', 'nélkülözhetetlen', 'alapvető', 'lényeges', 'elemi', 'lényegbeli', 'központi', 'középponti', 
                'esszenciális', 'elengedhetetlen', 'jelentős', 'releváns', 'jelentőségteljes', 'meghatározó', 'létfontosságú',
                'kulcsfontosságú', 'kardinális', 'hangsúlyos', 'kulcsponti', 'sarkalatos', 'lényegi', 'súlyponti', 'mérvadó', 
                'pótolhatatlan', 'fő', 'jelentékeny' ]

# These should also be removed as they are not in the context
STOPWORDS.update(QUESTION_WORDS)
STOPWORDS.update(URGENT_WORDS)

# Order the questions by importance (defined in the questions), and complexity
#

URGENT_PRIORITY = 1
COMPLEX_PRIORITY = 2
LONG_PRIORITY = 3
DEFAULT_PRIORITY = 4

def _question_priority(question: str) -> int:
    """Determine the priority of a question based on its content."""
    words = question.split()
    # Check for urgent or important keywords
    for word in words:
        if word.lower() in URGENT_WORDS:
            return URGENT_PRIORITY
    # Check for length and complexity (has , to separate multiple part of the question)
    if len(question) > 100 or "," in question:
        return LONG_PRIORITY
    if len(question) > 50:
        return COMPLEX_PRIORITY
    return DEFAULT_PRIORITY

def _filter(text: str) -> List[str]:
    """Filter out non-question sentences from a block of text."""
    # Split the text into sentences using regular expressions
    sentences = re.split('[.?!]', text)
    #sentences = re.split('[\n.?!]', text)
    questions = []
    for sentence in sentences:
        # clean the sentence
        sentence = sentence.strip().replace("\n", "")
        sentence = re.sub(' +', ' ', sentence)
        if sentence.endswith('?'):
            questions.append(sentence)
            continue
        words = sentence.split()
        for word in words:
            if word.lower() in QUESTION_WORDS:
                if not sentence.endswith('?'):
                    sentence += '?'
                questions.append(sentence)
                break
    return questions

def get_questions(text: str, top_k: int) -> List[str]:
    """Extract the top k questions from a block of text."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("Top k must be a positive integer.")
    try:
        questions = sorted(_filter(text), key=_question_priority)[:top_k]
        return questions
    except Exception as e:
        logging.exception(f"Error extracting questions: {e}")
        return []
#########################################################

def clean_text(texts):
    """
    Clean the given texts by removing stopwords and stemming the remaining words.

    :param texts: A list of strings representing the texts to be cleaned.
    :return: A list of strings representing the cleaned texts.
    """
    cleaned_texts = []
    
    def _proc_sentence(t):
        """
        Process a single sentence by tokenizing, stemming, and removing stopwords.

        :param t: A string representing a single sentence.
        :return: A list of strings representing the cleaned words in the sentence.
        """
        return [
            stemmer.stem(token.lower()) for token in word_tokenize(t)
            if token.lower() not in STOPWORDS and len(token) > 1
        ]
    for text in texts:
        cleaned_sentences = [_proc_sentence(t) for t in sent_tokenize(text)]
        cleaned_words = [word for sentence in cleaned_sentences for word in sentence]
        cleaned_texts.append(" ".join(cleaned_words))
    return cleaned_texts


def get_content():
    """
    Load the knowledge base from a PDF file, clean the text, and return the original and cleaned sentences.

    :return: A tuple containing a list of strings representing the original sentences and a list of strings representing
             the cleaned sentences.
    """
    logger.info("Loading knowledge base from PDF file...")
    all_pages_text = []
    with open(KNOWLEDGE_BASE, 'rb') as file:
        for page_layout in extract_pages(file):
            page_text = ""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text += element.get_text().strip() + " "
            all_pages_text.append(page_text)
    all_pages_text = " ".join(all_pages_text)

    logger.info("Cleaning text...")
    all_pages_text = re.sub(r'\n+', ' ', all_pages_text)
    all_pages_text = re.sub(r'\t+', ' ', all_pages_text)
    all_pages_text = re.sub(r'\s+', ' ', all_pages_text)
    sentences = sent_tokenize(all_pages_text)
    cleaned_sentences = clean_text(sentences)

    return sentences, cleaned_sentences

def get_relevant(question: List[str], sentences: List[str], cleaned_sentences: List[str]) -> List[str]:
    """"""
    
    """ 
    Get the relevant sentences from the context
    """
    try:
        # Vectorize sentences
        tfidf = TfidfVectorizer(tokenizer=word_tokenize, stop_words=list(STOPWORDS))
        tfidf.fit(cleaned_sentences)
        # Vectorize questions
        cleaned_question = clean_text(question)
        question_vector = tfidf.transform(cleaned_question)
 
        # Vectorize sentences and get the similarities
        sim_scores = cosine_similarity(tfidf.transform(cleaned_sentences), question_vector).ravel()
 
        # Select the top relevant sentences
        top_sentences = []
        for i in sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True):
            if len(top_sentences) < MAX_NUMBER_OF_RELEVANT_SENTENCES and sim_scores[i] > 0.1:
                top_sentences.append(sentences[i])

        return top_sentences

    except Exception as e:
        logger.error(f"An error occurred in get_relevant: {str(e)}")
        return []

def get_response(head: List[dict], questions: List[str], sentences: List[str], cleaned_sentences: List[str], history: List[dict]) -> List[dict]:
    """ Get the response from the ChatGPT API

    :param head (List[dict]): _description_
    :param questions (List[str]): _description_
    :param sentences (List[str]): _description_
    :param cleaned_sentences (List[str]): _description_
    :param history (List[dict]): _description_

    :return List[dict]: Retuns the response from the API
    """
    chatbot_response = []
    relevant_sentences = get_relevant(questions, sentences, cleaned_sentences)
    for question in questions:
        while True:
            try:
                message = head.copy()
                message[0]['content'] = message[0]['content'].replace('<knowledge_base>'," ".join(relevant_sentences))
                message.append({"role": "user", "content": f"Ha szerepel a tudásbázisban akkor válaszolj: {question}, de csak a Tudásbázis alapján válaszolhatsz!Mást ne is mondj!"})
                # Get the response from GPT-3
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=message,
                    max_tokens=2048,
                    stop=None,
                    temperature=0,
                    n=1  # how many answer to create?
                )
                break
            except Exception as e:
                print(f"Error occurred while getting response: {type(e).__name__} - {str(e)}")
                print("Clearing history")
                history.clear()
        response_text = response['choices'][0].message['content'].strip()
        chatbot_response.append({"role": "assistant", "content": f"{response_text}"})
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
      ]
    sentences , cleaned_sentences = get_content()
    logger.info('Sentence num in KnowledgeBase:',len(sentences))

    while True:
        #print("history:", history)
        user_input = input("User: ")
        if user_input == "exit":
            break
        chatbot_response = get_response( head, [user_input],sentences, cleaned_sentences, history)
        print(f"Chatbot:{chatbot_response[0]['content']}\n-----------------")

#
# WEBSEARCH mode
#

def google_search(query,num_of_results):
    # search not using googlesearch package
    query = urllib.parse.quote_plus(query)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    url = f'https://www.google.com/search?q={query}&num={num_of_results}'
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='g')
    
    links = []
    for result in search_results:
        link = result.find('a').get('href')
        if link.startswith('/url?q='):
            link = link[7:]
            link = link[:link.find('&sa=')]
            links.append(link)
    # Keep only real url ( not a pointer to internal)
    valid_links = [link for link in links if link != None and link.startswith('http')]
    # Filter out the searchengine related urls. 'google.com' 
    filtered_list = [url for url in valid_links if 'google.com' not in url ]

    return filtered_list

def bing_search(query,num_of_results) :
    query = urllib.parse.quote_plus(query)
     # The URL of Bing 
    url = f'https://www.bing.com/search?q={query}&rdr=1'
    response = requests.get(url)

    # Process the content
    soup = BeautifulSoup(response.text, 'html.parser')

    completeData = soup.find_all("li",{"class":"b_algo"})
    links= []
    for i in range(0, len(completeData)):
        link = completeData[i].find("a").get("href")
        links.append(link)
        print(link)

    # Keep only real url ( not a pointer to internal)
    valid_links = [l for l in links if l != None and l.startswith('http')]
    # Filter out the searchengine related urls. 'microsoft.com' & 'bing.com' 
    filtered_list = [url for url in valid_links if 'microsoft.com' not in url ] # and 'bing.com' not in url]

    return filtered_list

def get_webcontent(query,num_of_results):
    # Download search result
    try:
        # First try bing.com as goole limit the query from the same ip
        search_results = list(bing_search(query,num_of_results)) 
    except Exception as e :
        logging.error(e)
        try :
            # search using googlesearch package ! 
            search_results = list(search(query, lang='hu', num_results=num_of_results))
        except Exception as e :
            logging.error(e)
            return [] , []
    text = ""
    for result in search_results[:num_of_results]:
        print("Web content:",result)
        page = requests.get(result)
        soup = BeautifulSoup(page.content, 'html.parser')
        for element in soup.find_all(['p', 'h1', 'h2', 'h3']):
            text += element.text
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    #print(text)
    sentences = sent_tokenize(text)
    cleaned_sentences = clean_text(sentences)

    return sentences, cleaned_sentences


def websearch_mode():
    history = []
    head = [{"role": "system", 
            "content": "Legyél chatbot assistant \n\nTudásbázis:\n\n<knowledge_base>\n\n"},
      ]
    while True:
        #print("history:", history)
        user_input = input("User: ")
        if user_input == "exit":
            break
        sentences , cleaned_sentences = get_webcontent(user_input,MAX_URL_IN_WEBSEARCH)
        logger.info(f'Sentence num in webcontent: {len(sentences)}')

        chatbot_response = get_response( head, [user_input],sentences, cleaned_sentences, history)
        print(f"Chatbot:{chatbot_response[0]['content']}\n-----------------")




#
# MAIL_MODE
#

def read_unseen_mails(head: List, sentences: List, cleaned_sentences: List, history: List):
    # Connect to the mail server
    with imaplib.IMAP4_SSL(IMAP_HOST) as imap:
        imap.login(IMAP_USER, IMAP_PASS)
        imap.select('INBOX')

        # Download unseen mails
        status, messages = imap.search(None, 'UNSEEN')
        messages = messages[0].split(b' ')

        for mail in messages:
            # Download one mail
            if mail == b'' : # mail should be a number, the index to the mail. if empty then no mail at all
                continue
            res, msg = imap.fetch(mail, '(RFC822)')
            if res != 'OK' : # error in receiving (maybe deleted in the meanwile)
                continue
            for response in msg:
                if isinstance(response, tuple):
                    # Get the content and the headers
                    msg = email.message_from_bytes(response[1])
                    # decode the email subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        # if it's a bytes, decode to str
                        subject = subject.decode(encoding)
                    # decode email sender
                    sender, encoding = decode_header(msg.get("From"))[0]
                    if isinstance(sender, bytes):
                        sender = sender.decode(encoding)
                    # decode email recepient
                    recipient = msg['To']
                    recipient, encoding = decode_header(msg.get("To"))[0]
                    if isinstance(recipient, bytes):
                        recipient = recipient.decode(encoding)
                    
                    # Process the content
                    text = ""

                    if msg.is_multipart():
                            # iterate over email parts
                            for part in msg.walk():
                                # extract content type of email
                                content_type = part.get_content_type()
                                content_disposition = str(part.get("Content-Disposition"))
                                try:
                                    # get the email body
                                    body = part.get_payload(decode=True).decode()
                                except:
                                    pass
                                if content_type == "text/plain" and "attachment" not in content_disposition:
                                    # print text/plain emails and skip attachments
                                    text += body
                    else:
                            # extract content type of email
                            content_type = msg.get_content_type()
                            # get the email body
                            body = msg.get_payload(decode=True).decode()
                            if content_type == "text/plain":
                                text += body

                    # message_bytes tartalmazza az email szövegét bytes formátumban
                    questions = get_questions(text, top_k=MAX_QUESTION_TO_RESPOND)
                    chatbot_responses = []
                    for question in questions:
                        response = get_response(head, [question], sentences, cleaned_sentences, history)
                        chatbot_responses.append(response[0]['content'])

                    # Construct the email reply
                    reply = email.message.EmailMessage()
                    reply['From'] = recipient
                    reply['To'] = sender
                    reply['Subject'] = f"RE: {subject}"
                    s= "Válaszom a levelében feltett kérdésekre:\n\n"
                    for index, (k, v) in enumerate(zip(questions, chatbot_responses)):
                        s += f"{index+1}.K: {k}\n{index+1}.V: {v}\n\n"
                        
                    
                    #s = "\n".join(chatbot_responses)
                    reply.set_content(f"{s} \n\n Üdvözlettel\n Support\n---------------------------\n{text}\n\n",'utf-8')

                    # Send the reply via SMTP server
                    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                        server.starttls()
                        server.login(SMTP_USER, SMTP_PASS)
                        try:
                            server.send_message(reply)
                        except Exception as e:
                            print(e)
                    print(f"Válasz elküldve a következő címre: {sender}")

                    # Mark the mail as seen
                    imap.store(mail, "+FLAGS", "\\Seen")


def email_handler_mode():
    history = []
    head = [{"role": "system", 
            "content": "Csak a kérdésre válaszolj a Tudásbázis alapján!\n\Tudásbázis:\n\n<knowledge_base>\n\n"},
      ]
    sentences , cleaned_sentences = get_content()
    print('Sentence num:',len(sentences))
    while True:
       read_unseen_mails(head, sentences, cleaned_sentences, history)
       print("sleep")
       time.sleep(60)
 
#
# CONTROLLER mode
#
def controller_mode():

    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")


    head = [{"role": "system", 
            "content": f"""
Válaszolj a kérdésre olyan formában, hogy kérés az okos otthon vezérlőhöz JSON formátumban lesz elküldve. A kérések a következő 4 kategóriába sorolandók:
- "command": valamelyik kapcsoló állapotát változtatja meg. A JSON tartalmazza  következő propertiket: action, location, target, value, comment, scheduleTimeStamp.
- "query' : kérdezze le az adott eszköz állapotát. A JSON tartalmazza  következő propertiket: action, location, target, value, property.

A JSON válasz elemeinek magyarázata:
A válasz mindig egy lista lista
Az elemek :
Az "action" properti értékkészlete megegyezik a kérés kategóriájával: "command" , "query"
A "location" properti az adott szoba nevét tartalmazza kisbetűkkel
A "target" properti lehet: "szellőztető" ,"lámpa" , "termosztát" , hőmérő , redőny" kisbetűkkel
A "command" esetén a "scheduleTimeStamp" a mostani időhöz késleltetett időpontot tartalmazzon teljes dátum és idő megjelöléssel
Az okos otthon propertiei:
- helységek: konyha, nappali, hálószoba, fűrdő , folyosó, WC
- A WC villany bekapcsolása után 1 peccel kapcsolja be a "szellőztetőt" ezeket a parancsokat egy listában add
- A WC villany lekapcsolása után 3 perccel kapcsolja ki a "szellőztetőt"
- kapcsolni tudja minden helységben a villanykapcsolókat, le tudja kérdezni az állapotukat
- a helység hőmérsékletét a "hőmérő" lekérdezésével kapom.
- a helység hőmérsékletét a termosztát beállításával állítom.
Ha több parancsot kell kiadni akkor azt adja egy listában.
A pontos idő: {date_time}
A válaszod csak a JSON legyen semmi más
Minta: a wc villany bekapcsolása
[
{{
  "action": "command",
  "location": "WC",
  "target": "lámpa",
  "value": true,
  "comment": "WC villany felkapcsolása",
  "scheduleTimeStamp": "2023-09-03T12:18:00"
}},
 {{
  "action": "command",
  "location": "WC",
  "target": "szellőztető",
  "value": true,
  "comment": "WC villany bekapcsolása után 1 perccel kapcsolja be a szellőztetőt",
  "scheduleTimeStamp": "2023-09-03T12:18:59"
}}  
]
"""},
      ]
    while True:
        #print("history:", history)
        user_input = input("User: ")
        if not user_input.startswith("Controller"):
            continue
        if user_input == "exit":
            break
        try:
            message = head.copy()
            message.append({"role": "user", "content": user_input})
            # Get the response from GPT-3
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=message,
                max_tokens=2048,
                stop=None,
                temperature=0,
                n=1  # how many answer to create?
            )
        except Exception as e:
            print(f"Error occurred while getting response: {type(e).__name__} - {str(e)}")
        response_text = response['choices'][0].message['content'].strip()
        print(f"Controller:{response_text}\n-----------------")




if __name__ == "__main__":
 
    if MODE == "CHAT" :
        chat_mode()
    elif  MODE == "MAILHANDLER" :
        email_handler_mode()
    elif  MODE == "CONTROLLER" :
        controller_mode()
    elif  MODE == "WEBSEARCH" :
        websearch_mode()
    