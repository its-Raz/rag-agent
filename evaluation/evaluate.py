import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["OPENAI_API_KEY"] = ""
PROXY_BASE_URL = "https://api.llmod.ai/v1"
JUDGE_LLM_MODEL = "RPRTHPB-gpt-5-mini"
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"


judge_llm = ChatOpenAI(
    model=JUDGE_LLM_MODEL,
    base_url=PROXY_BASE_URL,
    temperature=1,
    api_key=os.environ["OPENAI_API_KEY"]
)

judge_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=PROXY_BASE_URL,
    api_key=os.environ["OPENAI_API_KEY"]
)



def call_rag_system(question: str):
    API_URL = "https://ted-talk-rag-agent-raz.vercel.app/api/prompt"
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            headers={"Content-Type": "application/json"},
            verify=False,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling API for question '{question}': {e}")
        return None



test_dataset = [
    {
        "question": "Find a TED talk that discusses how sports improve health. Provide the title and speaker.",
        "ground_truth": "The relevant talk is 'How playing sports benefits your body... and your brain' by speaker Jaspal Singh."
    },

]

print(" Starting Data Collection...")

data_samples = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}


for item in test_dataset:
    print(f"Processing: {item['question'][:50]}...")

    result = call_rag_system(item['question'])

    if result and 'response' in result:

        retrieved_texts = [ctx.get('chunk', '') for ctx in result.get('context', [])]


        if not retrieved_texts or all(not text.strip() for text in retrieved_texts):
            print(f"⚠️  Warning: No valid contexts found for question. Skipping.")
            continue


        answer = result['response'].strip()
        if not answer:
            print(f"⚠️  Warning: Empty answer received. Skipping.")
            continue


        print(f"   ✓ Answer length: {len(answer)} chars")
        print(f"   ✓ Contexts: {len(retrieved_texts)} chunks")
        print(f"   ✓ First context preview: {retrieved_texts[0][:100]}...")

        data_samples["question"].append(item['question'])
        data_samples["answer"].append(answer)
        data_samples["ground_truth"].append(item['ground_truth'])
        data_samples["contexts"].append(retrieved_texts)
    else:
        print(" Skipped question due to API error.")

if not data_samples["question"]:
    print(" No data collected. Exiting.")
    exit()

print(f"\n Collected {len(data_samples['question'])} samples")


print("\n Data structure check:")
for i, q in enumerate(data_samples["question"]):
    print(f"\nSample {i + 1}:")
    print(f"  Question: {q[:80]}...")
    print(f"  Answer: {data_samples['answer'][i][:80]}...")
    print(f"  Ground truth: {data_samples['ground_truth'][i][:80]}...")
    print(f"  Contexts count: {len(data_samples['contexts'][i])}")
    print(f"  Contexts type: {type(data_samples['contexts'][i])}")

rag_dataset = Dataset.from_dict(data_samples)

print("\n️  Running Ragas Evaluation...")
print("This may take a few minutes...\n")

metrics = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
]

try:
    results = evaluate(
        dataset=rag_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,

        raise_exceptions=True
    )
except Exception as e:
    print(f"\n Evaluation failed with error: {e}")
    print("\nTrying to get more details...")
    import traceback

    traceback.print_exc()
    exit()


df = results.to_pandas()


if 'question' not in df.columns:
    df['question'] = data_samples['question']
if 'answer' not in df.columns:
    df['answer'] = data_samples['answer']
if 'ground_truth' not in df.columns:
    df['ground_truth'] = data_samples['ground_truth']


wanted_columns = [
    "question",
    "answer",
    "ground_truth",
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevancy"
]

existing_columns = [col for col in wanted_columns if col in df.columns]
output_df = df[existing_columns]

pd.set_option('display.max_colwidth', 100)
print("\n === Detailed Report ===")
print(output_df)


print("\n Checking for NaN values:")
for col in output_df.columns:
    nan_count = output_df[col].isna().sum()
    if nan_count > 0:
        print(f"  ⚠️  {col}: {nan_count} NaN values")
    else:
        print(f"  ✓ {col}: No NaN values")


output_df.to_csv("rag_evaluation_results.csv", index=False)
print("\n Saved to 'rag_evaluation_results.csv'")

print("\n Global Average Scores:")
for metric in metrics:
    metric_name = metric.name
    if metric_name in df.columns:
        avg_score = df[metric_name].mean()
        if pd.isna(avg_score):
            print(f"  ️  {metric_name}: NaN (evaluation failed)")
        else:
            print(f"   {metric_name}: {avg_score:.4f}")