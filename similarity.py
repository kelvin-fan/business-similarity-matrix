# ============================================================
# Business Similarity Matrix
# Uses Claude (Anthropic) to generate business descriptions
# and OpenAI embeddings to compute pairwise cosine similarity
# across a user-defined list of companies.
#
# Inputs:  companies.txt  (one company name per line)
# Outputs: similarity_matrix.csv
#          similarity_matrix.png
# ============================================================

import os
import anthropic
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("companies.txt", "r") as f:
    COMPANIES = [line.strip() for line in f if line.strip()]

def get_business_description(company: str) -> str:
    """Call Claude to generate a concise business description for a company."""
    message = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Describe {company}'s core business in 3 sentences"
                f"Cover: what products or services they sell, who their customers are, "
                f"and what industry or market they compete in. Be factual and concise."
            )
        }]
    )
    return message.content[0].text

def get_embedding(text: str) -> np.ndarray:
    """Convert a text string into a 1536-dimensional embedding vector via OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(v1:np.ndarray, v2:np.ndarray) -> float:
    """Compute cosine similarity between two numpy vectors.
    Returns a value between -1 and 1.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    print("Step 1: Fetching business descriptions from Claude...")
    descriptions = {}
    for company in COMPANIES:
        print(f"  Describing {company}...")
        descriptions[company] = get_business_description(company)
        print(f"    → {descriptions[company][:80]}...")

    print("\nStep 2: Generating embedding vectors via OpenAI...")
    embeddings = {}
    for company, desc in descriptions.items():
        print(f"  Embedding {company}...")
        embeddings[company] = get_embedding(desc)

    print("\nStep 3: Computing similarity matrix...")
    n = len(COMPANIES)
    similarity_matrix = np.zeros((n, n))
    for i, c1 in enumerate(COMPANIES):
        for j, c2 in enumerate(COMPANIES):
            similarity_matrix[i][j] = cosine_similarity(embeddings[c1], embeddings[c2])

    print("\nStep 4: Saving outputs...")
    df = pd.DataFrame(similarity_matrix, index=COMPANIES, columns=COMPANIES)
    df.to_csv("similarity_matrix.csv", float_format="%.4f")
    print("  Saved similarity_matrix.csv")

    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="coolwarm", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(n), COMPANIES, rotation=45, ha="right", fontsize=10)
    plt.yticks(range(n), COMPANIES, fontsize=10)

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{similarity_matrix[i][j]:.2f}",
                    ha="center", va="center", fontsize=8, color="black")

    plt.title("Business Similarity Matrix", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("similarity_matrix.png", dpi=150)
    print("  Saved similarity_matrix.png")
    print("\nDone.")