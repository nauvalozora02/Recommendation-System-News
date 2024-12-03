import logging
from py2neo import Graph
import numpy as np
import pandas as pd

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Menghubungkan ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "komputer02"))

# Fungsi untuk menjalankan proses Node2Vec
def run_node2vec_process(graph):
    # Menghapus graf yang ada jika sudah ada
    logger.info("Menghapus graf yang ada jika sudah ada...")
    graph.run("""
    CALL gds.graph.exists('articleGraph')
    YIELD exists
    WITH exists 
    CALL gds.graph.drop('articleGraph', false) YIELD graphName
    WHERE exists
    RETURN graphName
    """)

    # Membuat graf baru dengan embeddings
    logger.info("Membuat graf baru dengan embeddings...")
    graph.run("""
    CALL gds.graph.project(
      'articleGraph',
      ['Article'],
      {
        TAGGED: {
          orientation: 'UNDIRECTED'
        }
      },
      {
        nodeProperties: ['embedding']
      }
    )
    """)

    # Menjalankan Node2Vec
    logger.info("Menjalankan Node2Vec...")
    result = graph.run("""
    CALL gds.node2vec.stream('articleGraph', {
      embeddingDimension: 64,
      walkLength: 5,
      returnFactor: 3.0,
      inOutFactor: 1.0
    })
    YIELD nodeId, embedding
    WITH nodeId, embedding
    MATCH (n:Article) WHERE id(n) = nodeId
    SET n.embedding = embedding
    RETURN n.id AS id, n.embedding AS embedding
    """).data()

    # Node2Vec selesai
    logger.info("Node2Vec selesai.")
    embeddings_list = []
    for record in result:
        # Normalisasi embedding
        embedding = np.array(record['embedding'])
        normalized_embedding = embedding / np.linalg.norm(embedding)
        graph.run("MATCH (n:Article {id: $id}) SET n.embedding = $embedding", id=record['id'], embedding=normalized_embedding.tolist())
        logger.info(f"Node ID: {record['id']}, Normalized Embedding: {normalized_embedding.tolist()}")
        embeddings_list.append({'id': record['id'], 'embedding': normalized_embedding.tolist()})
    
    # Simpan embeddings ke CSV
    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings.to_csv("Node2vec.csv", index=False)
    logger.info("Embeddings disimpan ke file 'Node2vec.csv'.")

# Menjalankan proses Node2Vec
run_node2vec_process(graph)
