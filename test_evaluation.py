"""
Test script to evaluate the RAG system with gold standard Q/A pairs
"""

import requests
import json
import time
import re
from typing import Dict, List

# Configuration
API_URL = "http://localhost:8000"

# Gold standard Q/A pairs
TEST_CASES = [
    {
        "id": 1,
        "question": "What is the average efficiency of modern solar panels?",
        "expected_keywords": ["15-22%", "23-24%", "efficiency", "premium"],
        "category": "factual"
    },
    {
        "id": 2,
        "question": "How do wind turbines compare to solar panels in terms of energy output per unit area?",
        "expected_keywords": ["2-3 megawatts", "150-200 watts", "square meter", "footprint"],
        "category": "comparative"
    },
    {
        "id": 3,
        "question": "Based on the document, what are the main barriers to renewable energy adoption and what solutions are proposed?",
        "expected_keywords": ["upfront costs", "intermittency", "infrastructure", "subsidies", "storage", "grid"],
        "category": "multi-hop"
    },
    {
        "id": 4,
        "question": "According to the document, what role does government policy play in renewable energy transition?",
        "expected_keywords": ["carbon pricing", "renewable energy mandates", "subsidies", "tax incentives", "research funding"],
        "category": "contextual"
    },
    {
        "id": 5,
        "question": "What is the chemical composition of hydrogen fuel cells?",
        "expected_keywords": [],
        "category": "no-answer"
    }
]

NO_ANSWER_PHRASES = [
    "not provided",
    "not included",
    "not covered",
    "insufficient information",
    "cannot be determined",
    "does not contain",
    "not available in the context"
]


def upload_document(file_path: str) -> Dict:
    print(f"\n📤 Uploading document: {file_path}")
    with open(file_path, "r") as f:
        text = f.read()

    response = requests.post(
        f"{API_URL}/upload",
        data={
            "text": text,
            "title": "Climate Change and Renewable Energy",
            "source": "test_document"
        }
    )
    result = response.json()
    print(f"✅ Upload successful: {result['chunks_created']} chunks created")
    return result


def query_system(question: str) -> Dict:
    response = requests.post(
        f"{API_URL}/query",
        json={"query": question, "top_k": 10, "rerank_top_n": 5}
    )
    return response.json()


def check_citations(answer: str) -> Dict:
    citations = re.findall(r"\[[\d,\s]+\]", answer)
    return {
        "has_citations": len(citations) > 0,
        "citation_count": len(citations)
    }


def is_correct_no_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in NO_ANSWER_PHRASES)


def evaluate_answer(test_case: Dict, result: Dict) -> Dict:
    answer = result["answer"]
    sources = result["sources"]
    citations = check_citations(answer)

    answer_lower = answer.lower()

    # Keyword evaluation (non–NO-ANSWER only)
    keywords_found = []
    keywords_missing = []

    for keyword in test_case["expected_keywords"]:
        if keyword.lower() in answer_lower:
            keywords_found.append(keyword)
        else:
            keywords_missing.append(keyword)

    keyword_recall = (
        len(keywords_found) / len(test_case["expected_keywords"])
        if test_case["expected_keywords"]
        else None
    )

    # Pass / fail logic
    if test_case["category"] == "no-answer":
        passed = is_correct_no_answer(answer)
        keyword_recall = 1.0 if passed else 0.0
    else:
        passed = (
            keyword_recall >= 0.6
            and citations["has_citations"]
            and len(sources) > 0
        )

    return {
        "test_id": test_case["id"],
        "category": test_case["category"],
        "passed": passed,
        "keyword_recall": keyword_recall,
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
        "citation_count": citations["citation_count"],
        "source_count": len(sources),
        "timing": result["timing"],
        "answer_length": len(answer)
    }


def run_evaluation():
    print("=" * 80)
    print("🧪 RAG SYSTEM EVALUATION")
    print("=" * 80)

    upload_document("docs/sample_document.txt")
    time.sleep(2)

    results = []

    for test_case in TEST_CASES:
        print(f"\n{'=' * 80}")
        print(f"📝 Test Case {test_case['id']}: {test_case['category'].upper()}")
        print(f"{'=' * 80}")
        print(f"Question: {test_case['question']}")

        start = time.time()
        result = query_system(test_case["question"])
        elapsed = time.time() - start

        evaluation = evaluate_answer(test_case, result)
        evaluation["total_query_time"] = elapsed
        results.append(evaluation)

        print(f"\n📊 Answer:\n{result['answer']}")
        print(f"\n✅ Citations: {evaluation['citation_count']}")
        print(f"✅ Sources: {evaluation['source_count']}")
        print(f"✅ Keyword Recall: {evaluation['keyword_recall']:.1%}")
        print(f"✅ Total Time: {elapsed:.2f}s")
        print(f"\n{'🎉 PASSED' if evaluation['passed'] else '❌ FAILED'}")

    # Summary
    passed = sum(r["passed"] for r in results)
    total = len(results)

    print(f"\n{'=' * 80}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n✅ Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")

    avg_recall = sum(r["keyword_recall"] for r in results) / total
    avg_time = sum(r["total_query_time"] for r in results) / total
    avg_citations = sum(
        r["citation_count"] for r in results if r["category"] != "no-answer"
    ) / (total - 1)

    print(f"📈 Average Keyword Recall: {avg_recall:.1%}")
    print(f"⏱️  Average Query Time: {avg_time:.2f}s")
    print(f"📚 Average Citations: {avg_citations:.1f}")

    with open("evaluation_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\n💾 Results saved to: evaluation_results.json")
    print("=" * 80)


if __name__ == "__main__":
    try:
        requests.get(f"{API_URL}/")
        print("✅ API is running")
    except Exception:
        print("❌ Backend not running")
        exit(1)

    run_evaluation()
