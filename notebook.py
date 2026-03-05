# %% [markdown]
# # Getting Started with Relevance Tuning
#
# Increasing recall by adding vectors using Jina embeddings and hybrid search.
# This notebook is based on the [Getting Started with Relevance Tuning](https://www.elastic.co/search-labs/blog/relevance-tuning-improving-recall-adding-vectors) Search Labs blog post.
#
# ## Pre-requisites
#
# - Python 3.10+
# - An Elasticsearch deployment with the Jina Embeddings v5 inference endpoint enabled
# - `ELASTICSEARCH_URL` and `ELASTICSEARCH_API_KEY` environment variables set (via `.env` file)

# %%
%pip install elasticsearch pandas plotly dotenv -q

# %% [markdown]
# ## Setup and Configuration

# %%
import os
import json
import pandas as pd
import plotly.graph_objects as go
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

es = Elasticsearch(ELASTICSEARCH_URL, api_key=ELASTICSEARCH_API_KEY)

INDEX_NAME = "ecommerce-products"

# %% [markdown]
# ## Inference Endpoint
# 
# Using Jina Embeddings v5 for semantic search.

# %%
INFERENCE_ENDPOINT_ID = ".jina-embeddings-v5-text-small"

print(f"Using predefined inference endpoint: {INFERENCE_ENDPOINT_ID}")

# %% [markdown]
# ## Create Index with Semantic Field

# %%
index_mappings = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "copy_to": "semantic_field"},
            "description": {"type": "text", "copy_to": "semantic_field"},
            "brand": {"type": "keyword"},
            "category": {"type": "keyword"},
            "semantic_field": {
                "type": "semantic_text",
                "inference_id": INFERENCE_ENDPOINT_ID,
            },
        }
    }
}

if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body=index_mappings)
    print(
        f"Created index: {INDEX_NAME} using inference endpoint: {INFERENCE_ENDPOINT_ID}"
    )
else:
    print(f"Index {INDEX_NAME} already exists")

# %% [markdown]
# ## Load and Prepare Dataset

# %%
import re

df = pd.read_csv("dataset.csv")


def first_sentence(text):
    """Extract the first sentence from a description."""
    text = str(text).strip()
    text = re.sub(r"^About this item\s*", "", text)
    match = re.match(r"([^.!?]+[.!?])", text)
    return match.group(1).strip() if match else text[:120]


def get_root_category(cats_str):
    """Extract root category from the categories list string."""
    try:
        import ast

        cats = ast.literal_eval(cats_str)
        return cats[0] if cats else "Other"
    except Exception:
        return "Other"


# Select and transform relevant fields
products = []
for _, row in df.iterrows():
    products.append(
        {
            "_id": row["asin"],
            "title": row["title"],
            "description": (
                first_sentence(row["description"])
                if pd.notna(row["description"])
                else ""
            ),
            "brand": row["brand"] if pd.notna(row["brand"]) else "",
            "category": get_root_category(row["categories"]),
        }
    )

print(f"Loaded {len(products)} products")
pd.DataFrame(products).head(10)

# %% [markdown]
# ## Index E-commerce Data

# %%
def bulk_index(products, index_name):
    """Bulk index products, using _id from the document."""
    actions = []
    for product in products:
        doc_id = product.get("_id")
        source = {k: v for k, v in product.items() if k != "_id"}

        action = {"_index": index_name, "_source": source}
        if doc_id:
            action["_id"] = doc_id
        actions.append(action)

    success, failed = helpers.bulk(es, actions, raise_on_error=False)

    if failed:
        print("Some documents failed to index:")
        for error in failed:
            print(f"  Error: {error}")
    else:
        print(f"Successfully indexed {success} documents")


bulk_index(products, INDEX_NAME)

# %% [markdown]
# ## Query Examples
# 
# Let's see how BM25 handles different types of queries.

# %%
def search_bm25(query):
    """Run a BM25 search and print results."""
    result = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "multi_match": {"query": query, "fields": ["title", "description"]}
            },
            "size": 5,
        },
    )
    print(f"Query: '{query}'")
    if not result["hits"]["hits"]:
        print("  No results found")
    for hit in result["hits"]["hits"]:
        print(
            f"  ID: {hit['_id']}, Score: {hit['_score']:.4f}, Title: {hit['_source']['title']}"
        )
    print()


# Query where BM25 works well (exact token match)
search_bm25("running shoes")

# Queries where BM25 struggles (intent-based, weak/no token overlap)
search_bm25("skincare routine")
search_bm25("study desk setup")
search_bm25("pet travel accessories")

# %% [markdown]
# ## Define Judgments for Evaluation
# 
# Creating relevance judgments for our top queries.

# %%
# Define judgments for rank_eval API
# Format: query_id, doc_id, grade (0=not relevant, 1=relevant, 2=highly relevant)
#
# Mix of queries:
# - q1: BM25-friendly (exact token match)
# - q2, q3, q4: intent-based (weak or no token overlap with documents)
judgments = [
    # Query 1: "running shoes" — BM25 handles well (tokens in title)
    {"query_id": "q1", "doc_id": "B09NQJFRW6", "grade": 2, "query": "running shoes"},  # Saucony Kinvara 13
    {"query_id": "q1", "doc_id": "B08JMD4LMM", "grade": 2, "query": "running shoes"},  # adidas Racer Tr21
    {"query_id": "q1", "doc_id": "B08VRJ6F2Q", "grade": 2, "query": "running shoes"},  # ASICS GT-1000 10
    {"query_id": "q1", "doc_id": "B07S8NRRWR", "grade": 2, "query": "running shoes"},  # adidas Ultraboost
    {"query_id": "q1", "doc_id": "B01HD620I8", "grade": 2, "query": "running shoes"},  # Salomon Speedcross 4
    {"query_id": "q1", "doc_id": "B07DX86321", "grade": 2, "query": "running shoes"},  # ASICS Gel-Cumulus 20
    {"query_id": "q1", "doc_id": "B0968YVLQ8", "grade": 1, "query": "running shoes"},  # Under Armour Charged Assert 9
    {"query_id": "q1", "doc_id": "B093QJ39ZS", "grade": 1, "query": "running shoes"},  # New Balance FuelCore Reveal
    {"query_id": "q1", "doc_id": "B096FGSC39", "grade": 1, "query": "running shoes"},  # Skechers GOrun Glide-Step
    {"query_id": "q1", "doc_id": "B01GVQWVV2", "grade": 1, "query": "running shoes"},  # ASICS GT-2000 7
    # Query 2: "skincare routine" — "routine" never appears in product titles
    {"query_id": "q2", "doc_id": "B08XMPKJ1L", "grade": 2, "query": "skincare routine"},  # Bio-Oil Skincare Body Oil Serum
    {"query_id": "q2", "doc_id": "B0BN3WQB92", "grade": 2, "query": "skincare routine"},  # Bare Peel Vitamin C Face Serum
    {"query_id": "q2", "doc_id": "B0BT7B7P5T", "grade": 2, "query": "skincare routine"},  # SkinInspired Retinol Face Serum
    {"query_id": "q2", "doc_id": "B00NPA2WEY", "grade": 2, "query": "skincare routine"},  # All Good Coconut Oil Moisturizer
    {"query_id": "q2", "doc_id": "B06XX6DS3P", "grade": 1, "query": "skincare routine"},  # Replenix Retinol Body Lotion
    {"query_id": "q2", "doc_id": "B07PDRD1KT", "grade": 1, "query": "skincare routine"},  # OxyGlow Charcoal Face Wash
    {"query_id": "q2", "doc_id": "B074J7869B", "grade": 1, "query": "skincare routine"},  # Greenberry Organics Face Wash
    {"query_id": "q2", "doc_id": "B08JV31QW4", "grade": 1, "query": "skincare routine"},  # LUXURIATE Tomato Face Wash
    {"query_id": "q2", "doc_id": "B00K3TVJMQ", "grade": 1, "query": "skincare routine"},  # Kaya Clinic Face Serum
    # Query 3: "study desk setup" — student context, products are desks/stands/pads
    {"query_id": "q3", "doc_id": "B08CS35J2T", "grade": 2, "query": "study desk setup"},  # Convenience Concepts Desk
    {"query_id": "q3", "doc_id": "B09B3LFDXJ", "grade": 2, "query": "study desk setup"},  # OImatser Monitor Stand Riser
    {"query_id": "q3", "doc_id": "B07W58LMND", "grade": 1, "query": "study desk setup"},  # YSAGi Desk Pad
    {"query_id": "q3", "doc_id": "B0CHYDX91L", "grade": 1, "query": "study desk setup"},  # LETTON Rose Gold Laptop Stand
    # Query 4: "pet travel accessories" — use-case grouping, no products say "travel accessories"
    {"query_id": "q4", "doc_id": "B08R8FRW53", "grade": 2, "query": "pet travel accessories"},  # CUBY Dog Sling Carrier
    {"query_id": "q4", "doc_id": "B01MYUYX33", "grade": 2, "query": "pet travel accessories"},  # GENORTH Dog Car Seats
    {"query_id": "q4", "doc_id": "B003C5RKE4", "grade": 2, "query": "pet travel accessories"},  # Precision Pet Wire Crate
    {"query_id": "q4", "doc_id": "B09GF8GBF6", "grade": 1, "query": "pet travel accessories"},  # Dog Seat Belt Set
    {"query_id": "q4", "doc_id": "B0CP3LQSWM", "grade": 1, "query": "pet travel accessories"},  # Portable Dog Water Bottle
]

judgments_df = pd.DataFrame(judgments)
print(judgments_df)

# %% [markdown]
# ## Lexical Search - BM25

# %%
# Create BM25 lexical search request
bm25_requests = []
for query_id, query_text in (
    judgments_df[["query_id", "query"]].drop_duplicates().values
):
    relevant_docs = judgments_df[judgments_df["query_id"] == query_id]
    ratings = [
        {"_index": INDEX_NAME, "_id": row["doc_id"], "rating": row["grade"]}
        for _, row in relevant_docs.iterrows()
    ]

    bm25_requests.append(
        {
            "id": query_id,
            "request": {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["title", "description"],
                    }
                }
            },
            "ratings": ratings,
        }
    )

# Quick test with recall metric
bm25_eval = {
    "requests": bm25_requests,
    "metric": {"recall": {"k": 10, "relevant_rating_threshold": 1}},
}

bm25_result = es.rank_eval(index=INDEX_NAME, body=bm25_eval)
print("BM25 Recall@10:", bm25_result.body["metric_score"])

# %% [markdown]
# ## Hybrid Search - BM25 + Vectors

# %%
# Hybrid search using RRF (Reciprocal Rank Fusion)
hybrid_requests = []
for query_id, query_text in (
    judgments_df[["query_id", "query"]].drop_duplicates().values
):
    relevant_docs = judgments_df[judgments_df["query_id"] == query_id]
    ratings = [
        {"_index": INDEX_NAME, "_id": row["doc_id"], "rating": row["grade"]}
        for _, row in relevant_docs.iterrows()
    ]

    hybrid_requests.append(
        {
            "id": query_id,
            "request": {
                "retriever": {
                    "rrf": {
                        "retrievers": [
                            {
                                "standard": {
                                    "query": {
                                        "multi_match": {
                                            "query": query_text,
                                            "fields": ["title", "description"],
                                        }
                                    }
                                }
                            },
                            {
                                "standard": {
                                    "query": {
                                        "match": {
                                            "semantic_field": {
                                                "query": query_text,
                                            }
                                        }
                                    }
                                }
                            },
                        ],
                        "rank_window_size": 50,
                        "rank_constant": 5,
                    }
                }
            },
            "ratings": ratings,
        }
    )

# Quick test with recall metric
hybrid_eval = {
    "requests": hybrid_requests,
    "metric": {"recall": {"k": 10, "relevant_rating_threshold": 1}},
}

hybrid_result = es.rank_eval(index=INDEX_NAME, body=hybrid_eval)
print("Hybrid Recall@10:", hybrid_result.body["metric_score"])

# %% [markdown]
# ## Results Comparison

# %%
# Compare recall across all search methods, both aggregate and per-query
methods = {
    "BM25 (Lexical)": bm25_requests,
    "Hybrid (BM25 + Vectors)": hybrid_requests,
}

recall_metric = {"recall": {"k": 10, "relevant_rating_threshold": 1}}

query_labels = dict(judgments_df[["query_id", "query"]].drop_duplicates().values)

comparison_data = []
per_query_data = []

for method_name, requests in methods.items():
    result = es.rank_eval(
        index=INDEX_NAME, body={"requests": requests, "metric": recall_metric}
    )
    comparison_data.append(
        {"method": method_name, "recall@10": result.body["metric_score"]}
    )
    for query_id, detail in result.body["details"].items():
        per_query_data.append(
            {
                "method": method_name,
                "query_id": query_id,
                "query": query_labels[query_id],
                "recall@10": detail["metric_score"],
            }
        )

comparison_df = pd.DataFrame(comparison_data)
per_query_df = pd.DataFrame(per_query_data)

print("Aggregate Recall Comparison:")
print(comparison_df.to_string(index=False))
print("\nPer-Query Recall:")
print(
    per_query_df.pivot(index="query", columns="method", values="recall@10").to_string()
)

# %% [markdown]
# ## Recall by Query
# 
# Breaking down recall per query shows where BM25 struggles and vectors help.

# %%
# Per-query recall breakdown
fig_per_query = go.Figure()

colors = {
    "BM25 (Lexical)": "#636EFA",
    "Hybrid (BM25 + Vectors)": "#00CC96",
}

for method_name in methods:
    method_data = per_query_df[per_query_df["method"] == method_name]
    fig_per_query.add_trace(
        go.Bar(
            x=method_data["query"].tolist(),
            y=method_data["recall@10"].tolist(),
            name=method_name,
            marker_color=colors[method_name],
        )
    )

fig_per_query.update_layout(
    title="Recall@10 by Query",
    xaxis_title="Query",
    yaxis_title="Recall@10",
    barmode="group",
    height=500,
)

fig_per_query.show(renderer="png")

# %% [markdown]
# ## Aggregate Recall Comparison

# %%
# Overall recall across methods
fig_aggregate = go.Figure()

fig_aggregate.add_trace(
    go.Bar(
        x=comparison_df["method"].tolist(),
        y=comparison_df["recall@10"].tolist(),
        name="Recall@10",
        marker_color=["#636EFA", "#00CC96"],
    )
)

fig_aggregate.update_layout(
    title="Recall@10: Lexical vs Hybrid",
    xaxis_title="Search Method",
    yaxis_title="Recall@10",
    height=500,
)

fig_aggregate.show(renderer="png")

# %% [markdown]
# ## Cleanup

# %%
es.indices.delete(index=INDEX_NAME)


