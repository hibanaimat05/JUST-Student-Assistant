import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = None
chunks = None
chunk_embeddings = None
tokenizer = None
model = None
RAG_THRESHOLD = 0.55
RAG_TOP_K = 3

def configure_rag(embedding_model_obj, chunks_obj, embeddings_obj, tokenizer_obj, model_obj, threshold=0.55, top_k=3):
    global embedding_model, chunks, chunk_embeddings, tokenizer, model, RAG_THRESHOLD, RAG_TOP_K
    embedding_model = embedding_model_obj
    chunks = chunks_obj
    chunk_embeddings = embeddings_obj
    tokenizer = tokenizer_obj
    model = model_obj
    RAG_THRESHOLD = threshold
    RAG_TOP_K = top_k

def retrieve(question, top_k=3):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score": float(scores[idx])
        })

    return results


def ask_just_assistant(question, top_k=3):

    results = retrieve(question, top_k=top_k)

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in results
    )

    messages = [
        {
            "role": "system",
            "content": """You are JUST Student Assistant.

Answer the user's question using ONLY the provided university context.

Rules:
1. Use only facts supported by the context.
2. Never invent phone numbers, emails, dates, or links.
3. If the answer is not available in the context, say:
   لا تتوفر لدي معلومات كافية للإجابة عن هذا السؤال.
4. Answer in the same language as the user's question.
5. Be clear and concise."""
        },
        {
            "role": "user",
            "content": f"""University Context:
{context}

Student Question:
{question}

Answer:"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return {
        "question": question,
        "answer": answer,
        "sources": [
            item["source"] for item in results
        ]
    }


def extract_facts(text):
    facts = {}

    patterns = {
        "fax": r"Fax\s*:\s*([^\n]+)",
        "phone": r"Phone\s*:\s*([^\n]+)",
        "email": r"Email\s*:\s*([^\s]+)",
        "url": r"https?://[^\s\])]+"
    }

    for key, pattern in patterns.items():
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )
        if matches:
            facts[key] = matches

    return facts


def detect_fact_type(question):
    question = question.lower().strip()

    if any(word in question for word in ["فاكس", "fax"]):
        return "fax"

    if any(word in question for word in [
        "هاتف", "تلفون", "رقم الهاتف", "phone"
    ]):
        return "phone"

    if any(word in question for word in [
        "إيميل", "ايميل",
        "بريد إلكتروني", "البريد الإلكتروني",
        "البريد الالكتروني", "email"
    ]):
        return "email"

    return None


def answer_contact_question(question, top_k=3):
    fact_type = detect_fact_type(question)

    if fact_type is None:
        return "لم أتمكن من تحديد نوع المعلومة المطلوبة."

    results = retrieve(question, top_k=top_k)

    for result in results:
        facts = extract_facts(result["text"])

        if fact_type in facts:
            return {
                "question": question,
                "answer": facts[fact_type][0],
                "fact_type": fact_type,
                "source": result["source"]
            }

    return "لم يتم العثور على المعلومة في المصادر المسترجعة."


def check_retrieval_relevance(question, top_k=3, threshold=0.60):
    results = retrieve(question, top_k=top_k)

    if not results:
        return "لا توجد مصادر مسترجعة."

    best_score = results[0]["score"]

    print("Question:", question)
    print("Best similarity score:", round(best_score, 4))
    print("Threshold:", threshold)

    if best_score < threshold:
        return "لا أملك معلومات كافية للإجابة عن هذا السؤال من قاعدة بيانات الجامعة."

    return "تم العثور على مصدر يحتمل أنه مرتبط بالسؤال."


def just_assistant(question):
    results = retrieve(question, top_k=RAG_TOP_K)

    if not results or results[0]["score"] < RAG_THRESHOLD:
        return {
            "answer": "لا أملك معلومات كافية للإجابة عن هذا السؤال من قاعدة بيانات الجامعة.",
            "source": None,
            "method": "Rejection"
        }

    # 1. Contact Extraction
    fact_type = detect_fact_type(question)

    if fact_type:
        for result in results:
            facts = extract_facts(result["text"])

            if fact_type in facts:
                return {
                    "answer": facts[fact_type][0],
                    "source": result["source"],
                    "method": "Structured Extraction"
                }

    # 2. Antivirus Extraction
    if any(word in question.lower() for word in [
        "مكافحة الفيروسات", "antivirus", "anti-virus"
    ]):
        for result in results:
            match = re.search(
                r"The university provides\s+(.+?)\s+software to",
                result["text"],
                flags=re.IGNORECASE
            )

            if match:
                return {
                    "answer": match.group(1).strip(),
                    "source": result["source"],
                    "method": "Structured Extraction"
                }

    # 3. Housing Extraction
    if any(word in question.lower() for word in [
        "سكن", "إسكان", "housing"
    ]):
        for result in results:
            steps_match = re.search(
                r"Application Steps:\s*(.*?)(?=Housing Department Contact:|$)",
                result["text"],
                flags=re.IGNORECASE | re.DOTALL
            )

            if steps_match:
                link_match = re.search(
                    r"https?://[^\s\])]+",
                    result["text"]
                )

                answer = steps_match.group(1).strip()

                if link_match:
                    answer += "\n\nApplication Link: " + link_match.group(0)

                return {
                    "answer": answer,
                    "source": result["source"],
                    "method": "Structured Extraction"
                }

    # 4. General RAG + Qwen
    rag_result = ask_just_assistant(question, top_k=RAG_TOP_K)

    return {
        "answer": rag_result["answer"],
        "source": ", ".join(
            dict.fromkeys(r["source"] for r in results)
        ),
        "method": "RAG + Qwen"
    }


