import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv

from memmachine.common.vector_graph_store import Node, Edge, Path
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import Neo4jVectorGraphStore
from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel

async def start(source, target):
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

    language_model = OpenAILanguageModel(
        {
            "model": "gpt-5-nano",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    )

    source_node = (await store.search_matching_nodes(
        required_properties={"name": source},
    ))[0]

    target_node = (await store.search_matching_nodes(
        required_properties={"name": target},
    ))[0]

    paths = await store.find_paths(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        max_hops=2,
    )

    path_representations = []
    for path in paths:
        path_representation = path.nodes[0].properties["name"]
        for i in range(len(path.edges)):
            edge = path.edges[i]
            node = path.nodes[i + 1]

            edge_representation = f"-[{edge.properties["relation"]}]-"
            if edge.source_uuid == node.uuid:
                edge_representation = f"<{edge_representation}"
            else:
                edge_representation = f"{edge_representation}>"

            path_representation += edge_representation + node.properties["name"]

        path_representations.append(path_representation)

    for path_representation in path_representations:
        print(path_representation)

    response_text, _ = await language_model.generate_response(
        system_prompt="Given the following paths between the two entities, determine if there is a meaningful connection between them. If so, explain the connection. If not, state that there is no meaningful connection.",
        user_prompt=f"Entity 1: {source}, Entity 2: {target}\n{'\n'.join(path_representations)}"
    )

    return response_text

def start_sync(source, target):
    load_dotenv()
    return asyncio.run(start(source, target))
