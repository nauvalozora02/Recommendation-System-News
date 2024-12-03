import re
import logging
from py2neo import Graph, Node, Relationship
import pandas as pd
from bs4 import BeautifulSoup
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from more_stopwords import more_stopword
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Menghubungkan ke Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "komputer02"))

# Fungsi untuk memuat data dari file JSON
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Fungsi untuk menghilangkan tag HTML dari teks
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Fungsi untuk proses case folding
def case_folding(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("-", " ")
    text = re.sub(r"\d+", "", text)
    return text

# Fungsi untuk menghilangkan stopwords
def remove_stopwords(text):
    stopword_factory = StopWordRemoverFactory()
    more_stopwords_list = more_stopword()
    stopwords = stopword_factory.get_stop_words() + more_stopwords_list
    return " ".join([word for word in text.split() if word not in stopwords])

# Fungsi untuk melakukan stemming
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# Fungsi untuk melakukan pra-pemrosesan pada teks
def preprocess_text(text):
    text = remove_html_tags(text)
    text = case_folding(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

# Fungsi untuk pra-pemrosesan dan pemberian bobot pada judul
def preprocess_and_weight_title(title, content, title_weight=3):
    combined_text = (title + " ") * title_weight + content
    preprocessed_text = preprocess_text(combined_text)
    return preprocessed_text

# Fungsi untuk membuat DataFrame dari data
def create_dataframe(data):
    df = pd.DataFrame.from_dict(data)
    return df[["_id", "original", "title", "content", "date", "kategori", "media", "relevan"]]

# Memuat data
file_path_news = 'pengujian_Sport/data_Uji_Berita_Baru_Sport.json'
file_path_history = 'pengujian_Sport/groundTruth_Riwayat_Berita_Sport.json'

data_news = load_data_from_json(file_path_news)
data_history = load_data_from_json(file_path_history)

# Membuat DataFrame
df_news = create_dataframe(data_news)
df_history = create_dataframe(data_history)

# Melakukan pra-pemrosesan pada DataFrame
df_news['preprocessed_content'] = df_news.apply(lambda x: preprocess_and_weight_title(x['title'], x['content']), axis=1)
df_history['preprocessed_content'] = df_history.apply(lambda x: preprocess_and_weight_title(x['title'], x['content']), axis=1)

# Menambahkan kolom 'type'
df_news["type"] = "new"
df_history["type"] = "history"

# Menggabungkan DataFrame
df_combined = pd.concat([df_news, df_history])

print(df_combined)

# Vektorisasi TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_combined["preprocessed_content"])
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Mengonversi matriks TF-IDF ke DataFrame untuk visualisasi
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df_combined._id)

# Menampilkan DataFrame TF-IDF
print(df_tfidf)

# Fungsi untuk menyimpan data ke Neo4j
from py2neo import Graph, Node, Relationship

def save_to_neo4j(df, tfidf_matrix, tfidf_feature_names):
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "komputer02"))
    
    for index, row in df.iterrows():
        tfidf_row = tfidf_matrix[index].toarray()[0].tolist()

        # Create Article node
        article_node = Node("Article", id=row["_id"], title=row["title"],
                            preprocessed_content=row["preprocessed_content"], date=row["date"],
                            kategori=row["kategori"], media=row["media"], type=row["type"],
                            embedding=tfidf_row, relevan=row["relevan"])
        graph.merge(article_node, "Article", "id")

        # Iterate over tfidf features
        for i, feature in enumerate(tfidf_feature_names):
            weight = float(tfidf_row[i])
            if weight > 0:
                tag_node = Node("Tag", name=feature)
                graph.merge(tag_node, "Tag", "name")
                
                # Create bidirectional TAGGED relationships
                tagged_rel_forward = Relationship(article_node, "TAGGED", tag_node, weight=weight)
                graph.merge(tagged_rel_forward)
                
                tagged_rel_backward = Relationship(tag_node, "TAGGED", article_node, weight=weight)
                graph.merge(tagged_rel_backward)

# Menyimpan data ke Neo4j dengan bobot TF-IDF
save_to_neo4j(df_combined, tfidf_matrix, tfidf_feature_names)

# Menyimpan DataFrame df_combined ke dalam file JSON
output_file_path = 'processed_data.json'  # Sesuaikan dengan path yang diinginkan
df_combined.to_json(output_file_path, orient='records', force_ascii=False)

# Menyimpan DataFrame TF_IDF ke dalam file CSV
output_file_path = 'TF-IDF.csv' 
df_tfidf.to_csv(output_file_path, index=True, encoding='utf-8')