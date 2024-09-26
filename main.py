import PyPDF2
import nltk
import spacy
import gensim
from gensim import corpora
from gensim.models.phrases import Phrases, Phraser
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
import re

# Load SpaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))
# Updating stop words with common debate-specific terms
stop_words.update([
    "inaudible", "would", "say", "audible", "president", "vice", "mr", "muir", 
    "thank", "thing", "know", "get", "want", "one", "take", "make", "davis"])

def read_pdf(file_path):
    """Reads and extracts text from a PDF file."""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def preprocess_text(text):
    """Lowercases, removes special characters, tokenizes, and lemmatizes text."""
    text = re.sub(r'[\n\r\t]', ' ', text)  # Remove newline and tab characters
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove special characters and digits
    doc = nlp(text)
    # Lemmatize and remove punctuation and digits   
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_digit]
    # Remove stop words
    return [word for word in lemmas if word not in stop_words and word != '-PRON-']
    

def segment_text_by_speaker(text):
    """Splits the text based on speaker patterns."""
    speaker_patterns = {
        'HARRIS': re.compile(r'\bHARRIS:\s*(.*?)\s*(?=\bTRUMP:|$)', re.DOTALL),
        'TRUMP': re.compile(r'\bTRUMP:\s*(.*?)\s*(?=\bHARRIS:|$)', re.DOTALL)
    }
    documents = [(speaker, segment.strip()) for speaker, pattern in speaker_patterns.items()
                 for segment in pattern.findall(text) if segment]
    return documents

def build_bigrams_and_trigrams(docs):
    """Creates bigrams and trigrams from the tokenized documents."""
    bigram = Phrases(docs, min_count=5, threshold=100)  # Higher threshold indicates fewer but more meaningful phrases
    trigram = Phrases(bigram[docs], threshold=100)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in docs]

def optimize_topic_count(corpus, dictionary, texts):
    """Find the optimal number of topics for LDA using coherence score."""
    max_score = -1
    best_model = None
    best_num_topics = 0
    for num_topics in range(5, 15):
        model = gensim.models.LdaModel(
            corpus=corpus, 
            id2word=dictionary, 
            num_topics=num_topics, 
            random_state=54, 
            chunksize=40,  
            passes=15,  
            iterations=100, 
            alpha='auto', 
            eta='auto'  
        )
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        if coherence_score > max_score:
            max_score = coherence_score
            best_model = model
            best_num_topics = num_topics
    return best_model, best_num_topics, max_score

def main(file_path):
    """Main function to run the LDA topic modeling process."""
    text = read_pdf(file_path)
    documents = segment_text_by_speaker(text)
    
    # Preprocess each document
    processed_docs = [preprocess_text(doc[1]) for doc in documents]
    
    # Build bigrams and trigrams
    processed_docs = build_bigrams_and_trigrams(processed_docs)
    
    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(processed_docs)
    # Filter out words that occur less than 5 documents, or more than 60% of the documents
    dictionary.filter_extremes(no_below=5, no_above=0.6)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # Find the optimal number of topics
    lda_model, best_num_topics, max_score = optimize_topic_count(corpus, dictionary, processed_docs)
    
    # Display the best topics
    for idx, topic in lda_model.print_topics(num_words=8):
        print(f"Topic: {idx} \nWords: {topic}")
    
    print('\nOptimal Number of Topics:', best_num_topics)
    print('Best Coherence Score:', max_score)

if __name__ == "__main__":
    file_path = "transcript_pres.pdf"  #  Path to transcript pdf file (can be changed)
    main(file_path)



