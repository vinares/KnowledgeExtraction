import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv

from memmachine.common.vector_graph_store import Node, Edge
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import Neo4jVectorGraphStore
from memmachine.common.embedder.openai_embedder import OpenAIEmbedder

async def start():
    store = Neo4jVectorGraphStore(
        {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        }
    )

    embedder = OpenAIEmbedder(
        {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    )

    node_map = {}
    nodes = []
    edges = []
    embedded_triple_texts = []
    with open("../schema_extraction/result/dev.txt") as triples:
        for triple in triples:
            source, target, relation = triple.strip().split("||")
            embedded_triple_text = f"({source})-[{relation}]->({target})"
            embedded_triple_texts.append(embedded_triple_text)

            if source not in node_map:
                node = Node(
                    uuid=uuid4(),
                    labels={"Entity"},
                    properties={"name": source}
                )
                nodes.append(node)
                node_map[source] = node.uuid

            if target not in node_map:
                node = Node(
                    uuid=uuid4(),
                    labels={"Entity"},
                    properties={"name": target}
                )
                nodes.append(node)
                node_map[target] = node.uuid

            edges.append(
                Edge(
                    uuid=uuid4(),
                    source_uuid=node_map[source],
                    target_uuid=node_map[target],
                    relation="RELATED_TO",
                    properties={"relation": relation, "triple_text": embedded_triple_text},
                )
            )

    embeddings = []
    for i in range(0, len(embedded_triple_texts), 100):
        batch = embedded_triple_texts[i:i+100]
        batch_embeddings = await embedder.ingest_embed(batch)
        embeddings.extend(batch_embeddings)

    for edge, embedding in zip(edges, embeddings):
        edge.properties["embedding"] = embedding

    await store.add_nodes(nodes)
    await store.add_edges(edges)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(start())