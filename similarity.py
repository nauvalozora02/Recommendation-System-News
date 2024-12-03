import logging
import pandas as pd
from py2neo import Graph
import json

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Menghubungkan ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "komputer02"))

# Menjalankan KNN
logger.info("Menjalankan KNN...")
graph.run("""
CALL gds.knn.write(
  'articleGraph',
  {
    nodeProperties: [{embedding: 'COSINE'}],
    topK: 5,
    sampleRate: 1.0,
    deltaThreshold: 0.001,
    maxIterations: 10,
    writeRelationshipType: 'KNN',
    writeProperty: 'score'
  }
)
""")
logger.info("Proses KNN selesai.")

# Mengambil hasil KNN dari Neo4j dengan kondisi tipe artikel
logger.info("Mengambil hasil KNN dari Neo4j...")
query = """
MATCH (n1:Article {type: 'history'})-[r:KNN]->(n2:Article {type: 'new'})
RETURN n1.id AS id1, n1.title AS title1, n1.kategori AS kategori1, n1.relevan AS relevan1, n1.type AS tipe1, n2.id AS id2, n2.title AS title2, n2.kategori AS kategori2, n2.relevan AS relevan2, n2.type AS tipe2, r.score AS similarity
ORDER BY id1, similarity DESC
"""
knn_results = graph.run(query).data()
logger.info("Hasil KNN telah diambil.")

# Mengkonversi hasil ke DataFrame dan normalisasi skor similarity
df_knn = pd.DataFrame(knn_results)
df_knn['similarity'] = df_knn['similarity'].astype(float)
max_similarity = df_knn['similarity'].max()
df_knn['similarity_normalized'] = df_knn['similarity'] / max_similarity

# Inisialisasi set untuk melacak ID artikel yang sudah digunakan
used_ids = set()

# Menyiapkan data JSON
json_data = []
for _, group in df_knn.groupby('id1'):
    filtered_group = []
    for idx, row in group.iterrows():
        if row['id2'] not in used_ids and row['tipe2'] == 'new':
            used_ids.add(row['id2'])
            filtered_group.append({
                "id": row["id2"],
                "title": row["title2"],
                "kategori": row["kategori2"],
                "relevan": row["relevan2"],
                "tipe": row["tipe2"],
                "similarity_score": row["similarity_normalized"]
            })
        if len(filtered_group) == 5:
            break
    if filtered_group:
        json_data.extend(filtered_group)

# Menyimpan hasil ke file JSON
with open("Rekomendasi.json", "w") as file:
    json.dump(json_data, file, indent=4)

logger.info("Hasil KNN 5 teratas hanya untuk tipe baru disimpan ke file knn_results.json.")
