# 1. Install dependencies
!pip install openai azure-cosmos langchain pymongo beautifulsoup4 requests tiktoken streamlit streamlit_jupyter nest_asyncio

# 2. Import libraries and set keys
import os
import streamlit as st
from streamlit_jupyter import StreamlitPatcher, tqdm
import nest_asyncio
import openai
import tiktoken
from pymongo import MongoClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
import hashlib
import uuid

# Enable Streamlit in Colab
nest_asyncio.apply()
StreamlitPatcher().jupyter()

# 3. Set API keys and database connection
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace
cosmos_url = "https://your-cosmos-account.documents.azure.com:443/"  # Replace
cosmos_key = "your-cosmos-key"
cosmos_db = "CreditOptimizerDB"
cosmos_container = "CardOffers"

# 4. Core functions
def estimate_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def extract_subject(question, model="gpt-3.5-turbo-0125"):
    prompt = [
        {"role": "system", "content": "Summarize this into 3-5 words."},
        {"role": "user", "content": f"'{question}'"}
    ]
    response = openai.ChatCompletion.create(
        model=model, messages=prompt, temperature=0, max_tokens=30
    )
    return response.choices[0].message["content"].strip()

def check_response_reasoning(question, response_json, model="gpt-3.5-turbo-0125"):
    messages = [
        {"role": "system", "content": "You are a financial reviewer. Check if the plan matches the user's goal logically."},
        {"role": "user", "content": f"Question: {question}\nResponse: {response_json}\nLogical? Explain briefly."}
    ]
    tokens = sum(estimate_tokens(m["content"], model=model) for m in messages)
    st.write(f"Token usage for review: {tokens} tokens (~${tokens * 5 / 1_000_000:.4f})")
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.4, max_tokens=400
    )
    return response.choices[0].message["content"]

cache_store = {}

def generate_cache_key(goal_string):
    return hashlib.sha256(goal_string.encode('utf-8')).hexdigest()

def query_credit_advisor_optimized(question, k=3):
    cache_key = generate_cache_key(question)
    if cache_key in cache_store:
        st.success("Cache hit: Reusing previous result!")
        return cache_store[cache_key]

    user_subject = extract_subject(question)
    st.write(f"Subject Inferred: **{user_subject}**")

    client = MongoClient("your-cosmos-mongo-uri")  # Replace with your real URI
    collection = client[cosmos_db][cosmos_container]
    embedding_model = OpenAIEmbeddings()
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model
    )
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content[:500] for doc in docs])

    if "simple cashback" in user_subject.lower() or "simple travel" in user_subject.lower():
        generation_model = "gpt-3.5-turbo-0125"
        st.info("Simple case detected: Using GPT-3.5-turbo to save cost.")
    else:
        generation_model = "gpt-4o"
        st.info("Complex case detected: Using GPT-4o for high quality.")

    role_prompt = (
        f"You are an expert on {user_subject}. Recommend a card optimization strategy "
        "based on context and user goal. Output must be compact JSON with card_plan, spending_strategy, redemption_plan."
    )
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Goal: {question}"}
    ]
    total_tokens = sum(estimate_tokens(m["content"], model=generation_model) for m in messages)
    st.write(f"Token usage for plan: {total_tokens} tokens (~${total_tokens * 5 / 1_000_000:.4f})")

    response = openai.ChatCompletion.create(
        model=generation_model,
        messages=messages,
        temperature=0.3,
        max_tokens=700
    )
    plan_json = response.choices[0].message["content"]

    review = check_response_reasoning(question, plan_json)

    cache_store[cache_key] = {
        "plan_json": plan_json,
        "review_notes": review
    }

    return {
        "plan_json": plan_json,
        "review_notes": review
    }

# 5. Streamlit UI
st.title("Credit Card Points Optimizer")

goal = st.text_input("Goal", value="100k Chase points by December")
timeline = st.text_input("Timeline", value="6 months")
spending = st.text_area("Monthly Spending", value="Groceries: $1000, Dining: $500")
cards = st.text_area("Current Cards", value="Freedom Unlimited")
prefs = st.text_area("Preferences", value="Italy trip, no Amex, Chase preferred")

submit = st.button("Generate Plan")

if submit:
    user_question = f"I want to {goal}. Timeline: {timeline}. Spending: {spending}. Current cards: {cards}. Preferences: {prefs}."
    result = query_credit_advisor_optimized(user_question)

    st.subheader("Generated Credit Card Plan")
    st.json(result["plan_json"])

    st.subheader("Self-Review and Reasoning")
    st.write(result["review_notes"])
