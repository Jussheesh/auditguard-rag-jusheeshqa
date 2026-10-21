import re
from typing import List, Dict, Any, Generator, Optional, Tuple
from app.llm import (
    get_openai_client, get_genai_client,
    load_image_bytes, make_genai_part, make_openai_image_part
)
from app.config import LLM_MODEL, USE_GOOGLE_GENAI, get_gemma_model

BASE_SYSTEM_PROMPT = """You are AuditGuard, a defense-grade document intelligence and compliance verification copilot.
Your job is to answer questions strictly and exclusively using the provided numbered source excerpts.

STRICT GROUNDING & MULTI-FILE SYNTHESIS RULES:
1. Every factual statement, metric, condition, or procedure MUST cite its supporting excerpt using in-line citation badges like [1], [2], etc. AND mention the source filename when multiple files are present.
2. MULTI-DOCUMENT SYNTHESIS: If excerpts from MULTIPLE different files (e.g., File A and File B) are provided below, you MUST synthesize and include findings from ALL relevant files. Never ignore an older file! If two files contain different numbers, dates, or rules on the same topic, explicitly compare both side-by-side and cite which file states what.
3. Directly answer the question in the opening sentence without conversational preambles (do NOT start with "Based on the provided excerpts", "According to document [1]", or "Sure").
4. For multi-part answers, metrics, penalties, or multi-file comparisons, format them into clean, executive bullet points with bold field/file names:
   - **File / Parameter:** Value and conditions [1]
   - **Comparison / Enforcement:** Specific requirement or consequence [2]
5. If the provided excerpts do not explicitly contain the answer, reply ONLY with:
   "I cannot verify this information in the provided document excerpts."
6. Do NOT invent, assume, extrapolate, or use outside unverified knowledge.
"""

def build_grounded_prompt(query: str, chunks: List[Dict[str, Any]], persona_addon: str = "") -> str:
    """Builds the context payload with numbered excerpts and explicit file inventory for Gemma 4."""
    context_lines = []
    unique_files = list(dict.fromkeys(c.get("filename", "Document") for c in chunks))
    files_header = f"FILES REPRESENTED IN EXCERPTS ({len(unique_files)}): " + ", ".join(unique_files)

    for c in chunks:
        cid = c.get("citation_id", 1)
        page = c.get("page", 1)
        doc = c.get("filename", "Document")
        parent = c.get("parent_text", "").strip()
        context_lines.append(f"[{cid}] (Source File: {doc} | Page/Slide/Sheet {page}):\n{parent}\n")

    joined_context = "\n".join(context_lines)

    multi_file_reminder = ""
    if len(unique_files) > 1:
        multi_file_reminder = (
            f"\nIMPORTANT: Excerpts come from {len(unique_files)} different files ({', '.join(unique_files)}). "
            f"You MUST synthesize information from ALL relevant files and explicitly name each file alongside its [citation_id]!"
        )

    return f"""=== VERIFIED SOURCE EXCERPTS ===
{files_header}

{joined_context}
================================

USER QUESTION:
{query}
{multi_file_reminder}
REMINDER: Answer strictly from the excerpts above. Cite [1], [2] directly on each claim and name the source file when comparing multiple documents. If unverified, refuse to guess."""

def clean_grounded_answer(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    if "I cannot verify this information" in cleaned:
        return cleaned

    preamble_patterns = [
        r'^(?:Based on the (?:provided )?(?:documents?|excerpts?|records?|context)[^,\n]*,\s*)',
        r'^(?:According to the (?:provided )?(?:documents?|excerpts?|records?|context)[^,\n]*,\s*)',
        r'^(?:Sure(?: thing)?[!,.]?\s*(?:Here (?:is|are)[^:\n]*:?\s*)?)',
        r'^(?:Certainly[!,.]?\s*(?:Here (?:is|are)[^:\n]*:?\s*)?)',
    ]
    for pat in preamble_patterns:
        cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE).strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    cleaned = re.sub(r'\[\s+(\d+)\s+\]', r'[\1]', cleaned)
    cleaned = re.sub(r'\s+([,.;])\s*\[(\d+)\]', r' [\2]\1', cleaned)
    cleaned = re.sub(r'^[•●▪◆]\s*', '- ', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()

def sanitize_citations(text: str, max_chunk: int) -> str:
    if max_chunk <= 0:
        return re.sub(r'\[\d+\]', '', text).strip()
    def _replace(m):
        val = int(m.group(1))
        if val > max_chunk or val <= 0:
            return ""
        return m.group(0)
    return re.sub(r'\[(\d+)\]', _replace, text).strip()

def verify_citations_grounding(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not answer:
        return {
            "all_supported": True,
            "verdict": "REFUSAL",
            "total_citations": 0,
            "unsupported_count": 0,
            "verified_citations": [],
            "reason": "Empty answer."
        }

    is_refusal = any(phrase in answer.lower() for phrase in [
        "cannot verify", "unverifiable", "refusal protocol", "out-of-bounds", 
        "no relevant text", "does not contain"
    ])
    if is_refusal:
        return {
            "all_supported": True,
            "verdict": "REFUSAL",
            "total_citations": 0,
            "unsupported_count": 0,
            "verified_citations": [],
            "reason": "Refusal answer contains no factual claims requiring citation."
        }

    if not chunks:
        return {
            "all_supported": False,
            "verdict": "NO_EVIDENCE",
            "total_citations": 0,
            "unsupported_count": 0,
            "verified_citations": [],
            "reason": "No evidence chunks provided to verify citations against."
        }

    chunk_map = {c.get("citation_id", idx + 1): c for idx, c in enumerate(chunks)}
    citation_positions = list(re.finditer(r'\[(\d+)\]', answer))
    
    if not citation_positions:
        return {
            "all_supported": False,
            "verdict": "NO_CITATIONS",
            "total_citations": 0,
            "unsupported_count": 0,
            "verified_citations": [],
            "reason": "Answer contains factual claims but completely lacks required citation badges [1], [2]."
        }

    verified_list = []
    unsupported_count = 0
    has_invalid_id = False
    has_unsupported_claim = False
    total_citations = 0

    prev_end = 0
    for match in citation_positions:
        cid = int(match.group(1))
        total_citations += 1
        start_idx = match.start()
        end_idx = match.end()

        clause_raw = answer[prev_end:start_idx].strip()
        sub_sentences = re.split(r'[.!?\n]+', clause_raw)
        claim_clause = sub_sentences[-1].strip() if sub_sentences else clause_raw
        claim_clause = re.sub(r'^[,\s]*(?:and|or|but|also|furthermore)?\s*', '', claim_clause, flags=re.IGNORECASE)
        prev_end = end_idx

        target_chunk = chunk_map.get(cid)
        if not target_chunk:
            unsupported_count += 1
            has_invalid_id = True
            verified_list.append({
                "citation_id": cid,
                "supported": False,
                "reason": f"Citation ID [{cid}] out of bounds (evidence pool size: {len(chunks)})",
                "claim_excerpt": claim_clause[:100]
            })
            continue

        chunk_text = (target_chunk.get("parent_text", "") + " " + target_chunk.get("child_text", "") + " " + target_chunk.get("filename", "")).lower()
        claim_text = claim_clause.lower()

        claim_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', claim_text)
        number_mismatch = False
        for num in claim_numbers:
            if len(num) > 1 and num not in chunk_text:
                number_mismatch = True
                break

        claim_tokens = [
            t for t in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', claim_text) 
            if t not in {
                "the", "and", "that", "this", "with", "from", "for", "are", "was", "were", 
                "been", "according", "records", "excerpts", "clause", "page", "section", "provides", "file", "source"
            }
        ]
        if claim_tokens:
            matched_tokens = [t for t in claim_tokens if t in chunk_text]
            overlap_ratio = len(matched_tokens) / len(claim_tokens)
        else:
            overlap_ratio = 1.0

        is_supported = (overlap_ratio >= 0.20) and not number_mismatch
        if not is_supported:
            unsupported_count += 1
            has_unsupported_claim = True

        verified_list.append({
            "citation_id": cid,
            "supported": is_supported,
            "overlap_score": round(overlap_ratio, 2),
            "number_verified": not number_mismatch,
            "claim_excerpt": claim_clause[:100],
            "reason": "Supported by evidentiary excerpt" if is_supported else (
                "Numerical discrepancy between clause and cited chunk" if number_mismatch 
                else "Insufficient textual support in cited excerpt"
            )
        })

    if has_invalid_id:
        verdict = "INVALID_CITATION"
    elif has_unsupported_claim:
        verdict = "UNSUPPORTED_CLAIM"
    elif total_citations == 0:
        verdict = "NO_CITATIONS"
    else:
        verdict = "VERIFIED"

    return {
        "all_supported": (unsupported_count == 0 and total_citations > 0),
        "verdict": verdict,
        "total_citations": total_citations,
        "unsupported_count": unsupported_count,
        "verified_citations": verified_list
    }

def extract_visual_images_from_chunks(chunks: List[Dict[str, Any]]) -> List[Tuple[bytes, str]]:
    visual_images: List[Tuple[bytes, str]] = []
    seen_urls = set()
    for c in chunks:
        urls = []
        if c.get("image_url"):
            urls.append(c.get("image_url"))
        for lv in c.get("linked_visuals", []):
            if lv.get("image_url"):
                urls.append(lv.get("image_url"))
        for u in urls:
            if u and u not in seen_urls:
                seen_urls.add(u)
                img_data = load_image_bytes(u)
                if img_data:
                    visual_images.append(img_data)
    return visual_images

def generate_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = "",
    model_override: Optional[str] = None
) -> Dict[str, Any]:
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)
    num_chunks = len(chunks)
    active_model = model_override or get_gemma_model()
    visual_images = extract_visual_images_from_chunks(chunks)

    if USE_GOOGLE_GENAI:
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                contents: List[Any] = []
                # Gemma 4 Best Practice: Place visual image parts BEFORE text prompt!
                for img_bytes, mime in visual_images:
                    part = make_genai_part(img_bytes, mime)
                    if part:
                        contents.append(part)
                contents.append(full_prompt)

                response = genai_cli.models.generate_content(
                    model=active_model,
                    contents=contents,
                )
                if response and response.text:
                    raw_text = response.text
                    citation_audit = verify_citations_grounding(raw_text, chunks)
                    clean_text = sanitize_citations(raw_text, num_chunks)
                    clean_text = clean_grounded_answer(clean_text)
                    actual_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        actual_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
                    
                    return {
                        "answer": clean_text,
                        "tokens": actual_tokens,
                        "is_fallback": False,
                        "model": active_model,
                        "has_multimodal_context": len(visual_images) > 0,
                        "citation_verification": citation_audit
                    }
            except Exception as e:
                print(f"[AuditGuard Google GenAI Error - Grounded Generator]: {e}")

    client = get_openai_client()
    try:
        if visual_images:
            user_parts: List[Dict[str, Any]] = []
            # Gemma 4 Best Practice: Image parts BEFORE text!
            for img_bytes, mime in visual_images:
                user_parts.append(make_openai_image_part(img_bytes, mime))
            user_parts.append({"type": "text", "text": user_content})
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_parts}
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

        response = client.chat.completions.create(
            model=active_model if model_override else LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600
        )
        content = response.choices[0].message.content or "No response from Gemma."
        citation_audit = verify_citations_grounding(content, chunks)
        clean_text = sanitize_citations(content, num_chunks)
        clean_text = clean_grounded_answer(clean_text)
        actual_tokens = getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") and response.usage else 0
        return {
            "answer": clean_text,
            "tokens": actual_tokens,
            "is_fallback": False,
            "model": active_model if model_override else LLM_MODEL,
            "has_multimodal_context": len(visual_images) > 0,
            "citation_verification": citation_audit
        }
    except Exception as e:
        print(f"[AuditGuard LLM Fallback Warning]: {e}")
        if chunks:
            # Multi-file aware extractive fallback: synthesize top chunks from each distinct file!
            seen_files = set()
            fallback_bullets = []
            for c in chunks:
                doc = c.get("filename", "Document")
                if doc not in seen_files or len(fallback_bullets) < 2:
                    seen_files.add(doc)
                    cid = c.get("citation_id", 1)
                    page = c.get("page", 1)
                    excerpt = c.get("parent_text", "").strip()
                    fallback_bullets.append(f"- **{doc} (Page/Slide/Sheet {page}):** {excerpt} [{cid}]")

            fallback_text = "\n\n".join(fallback_bullets) + "\n\n*(Verified across uploaded sources via AuditGuard Extractive Fallback)*"
            citation_audit = verify_citations_grounding(fallback_text, chunks)
            return {
                "answer": fallback_text,
                "tokens": 0,
                "is_fallback": True,
                "model": "AuditGuard Extractive Fallback",
                "has_multimodal_context": len(visual_images) > 0,
                "citation_verification": citation_audit
            }
        return {
            "answer": "I cannot verify this information in the provided document excerpts.",
            "tokens": 0,
            "is_fallback": True,
            "model": "AuditGuard Refusal",
            "has_multimodal_context": False,
            "citation_verification": {"all_supported": True, "total_citations": 0, "unsupported_count": 0, "verified_citations": []}
        }

def stream_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = "",
    model_override: Optional[str] = None
) -> Generator[str, None, None]:
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)
    active_model = model_override or get_gemma_model()
    visual_images = extract_visual_images_from_chunks(chunks)

    if USE_GOOGLE_GENAI:
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                contents: List[Any] = []
                for img_bytes, mime in visual_images:
                    part = make_genai_part(img_bytes, mime)
                    if part:
                        contents.append(part)
                contents.append(full_prompt)

                streamed_any = False
                for chunk in genai_cli.models.generate_content_stream(
                    model=active_model,
                    contents=contents,
                ):
                    if chunk.text:
                        streamed_any = True
                        yield chunk.text
                if streamed_any:
                    return
            except Exception as e:
                print(f"[AuditGuard Google GenAI Error - Streaming Generator]: {e}")

    client = get_openai_client()
    try:
        if visual_images:
            user_parts: List[Dict[str, Any]] = []
            for img_bytes, mime in visual_images:
                user_parts.append(make_openai_image_part(img_bytes, mime))
            user_parts.append({"type": "text", "text": user_content})
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_parts}
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

        stream = client.chat.completions.create(
            model=active_model if model_override else LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    except Exception:
        if chunks:
            seen_files = set()
            for c in chunks:
                doc = c.get("filename", "Document")
                if doc not in seen_files:
                    seen_files.add(doc)
                    cid = c.get("citation_id", 1)
                    page = c.get("page", 1)
                    excerpt = c.get("parent_text", "").strip()
                    yield f"- **{doc} (Page {page}):** {excerpt} [{cid}]\n\n"
        else:
            yield "I cannot verify this information in the provided document excerpts."