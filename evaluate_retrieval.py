from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Same configuration used by retriever.py
CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3

# 10 evaluation questions with their dummy expected ticket IDs for testing
TEST_CASES = [
    {
        "question": "My calls keep dropping during conversations. How can I fix this?",
        "expected_ticket_id": "TK-002"
    },
    {
        "question": "My mobile internet is very slow even though I have a good signal.",
        "expected_ticket_id": "TK-003"
    },
    {
        "question": "I cannot connect to mobile data after inserting a new SIM card.",
        "expected_ticket_id": "TK-004"
    },
    {
        "question": "How can I activate international roaming on my number?",
        "expected_ticket_id": "TK-005"
    },
    {
        "question": "I am unable to make outgoing calls from my phone.",
        "expected_ticket_id": "TK-006"
    },
    {
        "question": "My phone shows SIM not detected even though the SIM is inserted.",
        "expected_ticket_id": "TK-007"
    },
    {
        "question": "Why is my monthly mobile bill higher than usual?",
        "expected_ticket_id": "TK-008"
    },
    {
        "question": "I am not receiving OTP messages on my mobile number.",
        "expected_ticket_id": "TK-009"
    },
    {
        "question": "My network signal is very weak inside my house.",
        "expected_ticket_id": "TK-010"
    },
    {
        "question": "I recharged my phone but the data benefits are not showing up.",
        "expected_ticket_id": "TK-011"
    }
]


def evaluate_retrieval():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    correct = 0
    total = len(TEST_CASES)

    print(f"\nEvaluating ticket retrieval (Recall@{TOP_K})")
    print("=" * 70)

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]
        expected_id = str(test_case["expected_ticket_id"])

        # Retrieve top 3 tickets
        results = tickets_store.similarity_search_with_score(
            question,
            k=TOP_K
        )

        retrieved_ids = []

        for doc, distance in results:
            ticket_id = doc.metadata.get("ticket_id")

            if ticket_id is not None:
                retrieved_ids.append(str(ticket_id))

        # Check whether expected ticket is in top 3
        is_correct = expected_id in retrieved_ids

        if is_correct:
            correct += 1

        print(f"\nTest case {index}:")
        print(f"Question       : {question}")
        print(f"Expected ticket: {expected_id}")
        print(f"Retrieved IDs  : {retrieved_ids}")
        print(f"Result         : {'PASS' if is_correct else 'FAIL'}")

    recall_at_k = correct / total if total else 0

    print("\n" + "=" * 70)
    print(f"Correctly retrieved: {correct}/{total}")
    print(f"Recall@{TOP_K}      : {recall_at_k:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_retrieval()