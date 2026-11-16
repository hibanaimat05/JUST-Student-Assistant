
import re
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = None
chunks = None
chunk_embeddings = None
tokenizer = None
model = None

RAG_THRESHOLD = 0.55
RAG_TOP_K = 3

FALLBACK_ANSWER = (
    "لا تتوفر لدي معلومات كافية للإجابة عن هذا السؤال "
    "من قاعدة بيانات الجامعة."
)


def configure_rag(
    embedding_model_obj,
    chunks_obj,
    embeddings_obj,
    tokenizer_obj,
    model_obj,
    threshold=0.55,
    top_k=3
):
    global embedding_model, chunks, chunk_embeddings
    global tokenizer, model, RAG_THRESHOLD, RAG_TOP_K

    if embedding_model_obj is None:
        raise ValueError("Embedding model is missing.")

    if not chunks_obj:
        raise ValueError("Knowledge base chunks are empty.")

    if embeddings_obj is None:
        raise ValueError("Chunk embeddings are missing.")

    if tokenizer_obj is None or model_obj is None:
        raise ValueError("Tokenizer or language model is missing.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    embeddings = np.asarray(embeddings_obj)

    if embeddings.ndim != 2:
        raise ValueError("Chunk embeddings must be a 2D array.")

    if len(chunks_obj) != len(embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    embedding_model = embedding_model_obj
    chunks = chunks_obj
    chunk_embeddings = embeddings
    tokenizer = tokenizer_obj
    model = model_obj

    RAG_THRESHOLD = threshold
    RAG_TOP_K = top_k

    return True


def validate_rag_configuration():
    if any(
        item is None
        for item in [
            embedding_model,
            chunks,
            chunk_embeddings,
            tokenizer,
            model
        ]
    ):
        raise RuntimeError(
            "RAG is not configured. Call configure_rag() first."
        )

    if len(chunks) == 0:
        raise RuntimeError("Knowledge base is empty.")


def retrieve(question, top_k=None):
    validate_rag_configuration()

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    top_k = RAG_TOP_K if top_k is None else top_k

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(query_embedding)

    if query_embedding.ndim != 2:
        raise ValueError("Invalid query embedding shape.")

    if query_embedding.shape[1] != chunk_embeddings.shape[1]:
        raise ValueError("Query and document embedding dimensions differ.")

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


def get_relevant_results(question, top_k=None):
    results = retrieve(question, top_k=top_k)

    return [
        item for item in results
        if item["score"] >= RAG_THRESHOLD
    ]


def ask_just_assistant(question, top_k=None):
    results = get_relevant_results(question, top_k=top_k)

    if not results:
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": []
        }

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in results
    )

    messages = [
        {
            "role": "system",
            "content": """You are JUST Student Assistant.

Answer using ONLY the provided university context.

Rules:
1. Use only facts supported by the context.
2. Never invent phone numbers, emails, dates, or links.
3. If the answer is unavailable, say:
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
        "sources": list(dict.fromkeys(
            item["source"] for item in results
        ))
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

    # Password-reset questions must not trigger contact extraction
    reset_terms = [
        "password", "reset", "forgot",
        "كلمة المرور", "كلمة السر",
        "إعادة تعيين", "نسيت"
    ]

    if any(term in question for term in reset_terms):
        return None

    if any(word in question for word in ["فاكس", "fax"]):
        return "fax"

    if any(word in question for word in [
        "هاتف", "تلفون", "رقم الهاتف", "phone"
    ]):
        return "phone"

    contact_terms = [
        "email address", "contact email",
        "what is the email", "email of",
        "إيميل", "ايميل", "البريد الإلكتروني",
        "البريد الالكتروني", "عنوان البريد"
    ]

    if any(term in question for term in contact_terms):
        return "email"

    return None


def answer_contact_question(question, top_k=None):
    fact_type = detect_fact_type(question)

    if fact_type is None:
        return {
            "answer": "لم أتمكن من تحديد نوع المعلومة المطلوبة.",
            "source": None
        }

    results = get_relevant_results(question, top_k=top_k)

    for result in results:
        facts = extract_facts(result["text"])

        if fact_type in facts:
            return {
                "question": question,
                "answer": facts[fact_type][0],
                "fact_type": fact_type,
                "source": result["source"]
            }

    return {
        "question": question,
        "answer": FALLBACK_ANSWER,
        "source": None
    }


def check_retrieval_relevance(
    question,
    top_k=None,
    threshold=None
):
    results = retrieve(question, top_k=top_k)

    if not results:
        return "لا توجد مصادر مسترجعة."

    threshold = RAG_THRESHOLD if threshold is None else threshold
    best_score = results[0]["score"]

    print("Question:", question)
    print("Best similarity score:", round(best_score, 4))
    print("Threshold:", threshold)

    if best_score < threshold:
        return FALLBACK_ANSWER

    return "تم العثور على مصدر يحتمل أنه مرتبط بالسؤال."


def just_assistant(question):
    results = get_relevant_results(question, top_k=RAG_TOP_K)

    if not results:
        return {
            "answer": FALLBACK_ANSWER,
            "source": None,
            "method": "Rejection"
        }

    fact_type = detect_fact_type(question)

    # Contact information extraction
    if fact_type:
        for result in results:
            facts = extract_facts(result["text"])

            if fact_type in facts:
                return {
                    "answer": facts[fact_type][0],
                    "source": result["source"],
                    "method": "Structured Extraction"
                }

    # Antivirus information extraction
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

    # Housing information extraction
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

    # General RAG + Qwen
    rag_result = ask_just_assistant(
        question,
        top_k=RAG_TOP_K
    )

    return {
        "answer": rag_result["answer"],
        "source": ", ".join(rag_result["sources"]) or None,
        "method": "RAG + Qwen"
    }
