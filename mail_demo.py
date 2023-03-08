import warnings

warnings.filterwarnings('ignore')

import email
import imaplib
import logging
import os
import re
import smtplib
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()

import nltk
from nltk.stem.snowball import SnowballStemmer
#from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('stopwords')
stemmer = SnowballStemmer("hungarian")
STOPWORDS = set(nltk.corpus.stopwords.words('hungarian'))


import openai
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
MAX_QUESTION_TO_RESPOND = int(os.getenv("MAX_QUESTION_TO_RESPOND"))
MAX_NUMBER_OF_RELEVANT_SENTENCES = int(os.getenv("MAX_NUMBER_OF_RELEVANT_SENTENCES"))
MODEL = "gpt-3.5-turbo"


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
# Extract question
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
#
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
    sentences = re.findall('[^.?!]+[.?!]', text)
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
        cleaned_texts.append([_proc_sentence(t) for t in sent_tokenize(text)])
    cleaned_texts = [" ".join(item) for sublist in cleaned_texts for item in sublist]     
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
            res, msg = imap.fetch(mail, '(RFC822)')

            for response in msg:
                if isinstance(response, tuple):
                    # Get the content and the headers
                    msg = email.message_from_bytes(response[1])
                    sender = msg['From']
                    recipient = msg['To']
                    subject = msg['Subject']

                    # Process the content
                    text = msg.get_payload()
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
                    s = "\n".join(chatbot_responses)
                    reply.set_content(f"Válaszom a levelére:\n{s}\n\nÜdvözlettel,\nSupport")

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
 


if __name__ == "__main__":
 
    if MODE == "CHAT" :
        chat_mode()
    else :
        email_handler_mode()
