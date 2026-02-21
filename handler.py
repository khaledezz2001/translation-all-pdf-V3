import time
import re
import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Single model path — OpenPipe/Qwen3-14B-Instruct
# =====================================================
MODEL_PATH = "/models/hf/qwen"

model_tokenizer = None
model = None

# =====================================================
# Default system prompts
# =====================================================
DEFAULT_SUMMARY_PROMPT = (
    "You are a professional legal assistant.\n"
    "Produce a single-paragraph summary of the ENTIRE document in clear English.\n"
    "STRICT RULES:\n"
    "- Output MUST be one paragraph only\n"
    "- Do NOT use headings, titles, bullet points, or lists\n"
    "- Do NOT classify the document type unless explicitly stated in the text\n"
    "- Do NOT invent or infer information\n"
    "- Mention only facts that are explicitly present in the document\n"
    "- Cover all major sections evenly if the document is long\n"
    "- Focus on parties, purpose, key obligations, payments, terms, penalties, and dispute resolution if present\n"
    "- Ignore layout, tables, formatting, and section numbering\n"
    "- Write in neutral legal English\n\n"
)

TRANSLATE_SYSTEM_PROMPT = (
    "You are a certified legal translator specializing in Russian-to-English legal documents.\n"
    "Translate the following text from Russian to English.\n"
    "STRICT RULES:\n"
    "- Translate ONLY — do NOT summarize, paraphrase, or add commentary\n"
    "- Preserve the original meaning, tone, and structure as closely as possible\n"
    "- Keep proper nouns, names, dates, and numbers unchanged\n"
    "- Preserve paragraph breaks and line structure\n"
    "- If a word or phrase is already in English, keep it as-is\n"
    "- Output ONLY the English translation, nothing else\n"
    "TRANSLITERATION RULES (CRITICAL):\n"
    "- Use phonetic transliteration that matches the Cyrillic spelling letter-by-letter\n"
    "- Complete mapping: А→A, Б→B, В→V, Г→G, Д→D, Е→E, Ё→Yo, Ж→Zh, З→Z, "
    "И→I, Й→Y, К→K, Л→L, М→M, Н→N, О→O, П→P, Р→R, С→S, Т→T, "
    "У→U, Ф→F, Х→Kh, Ц→Ts, Ч→Ch, Ш→Sh, Щ→Shch, Ъ→(omit), Ы→Y, Ь→(omit), Э→E, Ю→Yu, Я→Ya\n"
    "- 'Кс' → 'Ks' (NEVER 'X'). Example: Ксенофонтов → Ksenofontov (NOT Xenofontov)\n"
    "- 'Е' → always 'E' (NEVER 'I'). Example: ГРИСЕН → GRISEN (NOT GRISIN)\n"
    "- Company names in Cyrillic are phonetic transcriptions — transliterate them back faithfully\n"
    "LEGAL TRANSLATION STANDARDS:\n"
    "- Use formal legal English register: use 'shall' for obligations, 'hereby' for declarations\n"
    "- Preserve civil law terminology: translate 'Цедент' as 'Cedent' (NOT 'Assignor'), "
    "'Цессионарий' as 'Cessionary' (NOT 'Assignee')\n"
    "- Translate 'Устав' as 'Articles of Association' (NOT 'Charter')\n"
    "- Translate 'Договор цессии' as 'Conveyance Agreement' or 'Cession Agreement'\n"
    "- Translate 'перевод долга' as 'debt transfer' (NOT 'debt assignment')\n"
    "- Translate 'Договор о совместной деятельности' as 'Joint Operation Agreement' (NOT 'Joint Venture Agreement')\n"
    "- Use standard English date format: 'October __, 2025' (NOT '_ October 2025')\n"
    "- Maintain formal legal phrasing: 'represented by its director', 'acting under'\n"
    "- Use 'such as' instead of 'for example' in legal clauses\n"
    "- Use 'shall be' and 'shall become' for future obligations\n"
)


# =====================================================
# Load single Qwen3-14B model
# =====================================================
def load_model():
    global model_tokenizer, model
    if model is not None:
        return

    log("Loading OpenPipe/Qwen3-14B-Instruct (FP16)")

    model_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    model.eval()

    # Enable CUDA optimizations for H100
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log(f"Model loaded on device: {model.device}")

# =====================================================
# Detect layout separators
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

# =====================================================
# TRANSLATION — Using Qwen 14B
# =====================================================
def translate_text(text: str) -> str:
    """Translate Russian text to English using Qwen 14B."""
    if not text or not text.strip():
        return text

    # Split text into manageable chunks (by paragraphs/lines)
    # to avoid exceeding context length
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        line_len = len(line.split())
        # Keep chunks under ~1500 words to leave room for translation output
        if current_len + line_len > 1500 and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = line_len
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    translated_chunks = []

    for chunk_idx, chunk in enumerate(chunks):
        stripped = chunk.strip()

        # Skip empty chunks
        if not stripped:
            translated_chunks.append(chunk)
            continue

        # Skip chunks with too few letter characters (likely just numbers/symbols)
        if len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 5:
            translated_chunks.append(chunk)
            continue

        log(f"Translating chunk {chunk_idx + 1}/{len(chunks)} ({len(stripped.split())} words)")

        messages = [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": stripped}
        ]

        try:
            prompt = model_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = (
                f"<|im_start|>system\n{TRANSLATE_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{stripped}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = model_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=16384  # Qwen3 supports 32K native context
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=model_tokenizer.pad_token_id,
                eos_token_id=model_tokenizer.eos_token_id
            )

        # Decode only new tokens
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        decoded = model_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean up think tags and any remaining special tokens
        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
        decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()

        # Remove echoed system prompt if model regurgitated instructions
        for marker in ["STRICT RULES:", "LEGAL TRANSLATION STANDARDS:"]:
            if marker in decoded:
                idx = decoded.find(marker)
                after = decoded[idx:]
                lines = after.split("\n")
                # Find last instruction-like line (starts with "- ")
                last_rule = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("- "):
                        last_rule = i
                # Keep only content after the instruction block
                decoded = "\n".join(lines[last_rule + 1:]).strip()

        translated_chunks.append(decoded)

    return "\n".join(translated_chunks)

# =====================================================
# OCR cleanup
# =====================================================
def clean_ocr_noise(text: str) -> str:
    cleaned = []
    seen = set()

    for raw in text.split("\n"):
        line = raw.strip()
        upper = line.upper()

        if not line:
            continue
        if is_layout_line(line):
            continue
        if len(re.findall(r"[A-Za-z]", line)) < 5:
            continue
        if upper in seen:
            continue

        seen.add(upper)
        cleaned.append(line)

    return "\n".join(cleaned)

# =====================================================
# Word limiter
# =====================================================
def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    # Take max_words, then try to find the last sentence-ending punctuation
    truncated = " ".join(words[:max_words])

    # Look for the last sentence boundary (., !, ?)
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    last_excl = truncated.rfind("! ")
    last_quest = truncated.rfind("? ")

    # Also check if the truncated text ends with a period
    if truncated.rstrip().endswith("."):
        return truncated.rstrip()

    best = max(last_period, last_excl, last_quest)

    # If we found a sentence boundary in the last 40% of the text, cut there
    if best > len(truncated) * 0.6:
        return truncated[:best + 1].strip()

    # Otherwise just return the truncated text
    return truncated.rstrip()

# =====================================================
# SUMMARY — Using Qwen 14B
# =====================================================
def summarize_all_pages(pages, max_words: int, system_prompt: str):
    # Combine all pages
    full_text = "\n\n".join(
        cleaned
        for p in pages
        if (cleaned := clean_ocr_noise(p["text"]))
        and len(re.findall(r"[A-Za-z]", cleaned)) > 20
    )

    if not full_text.strip():
        log("ERROR: No valid text found for summary")
        return ""

    doc_word_count = len(full_text.split())
    log(f"Full text length: {len(full_text)} chars, {doc_word_count} words")

    # Smart word limit: scale target based on document length
    # Prevents hallucination on short documents
    actual_target = max(50, min(max_words, doc_word_count // 3))
    if actual_target < max_words:
        log(f"Smart limit: requested {max_words} words, but document is only {doc_word_count} words → target adjusted to {actual_target}")
    else:
        log(f"Target: {actual_target} words (document is long enough)")

    # Build messages for Qwen with word count instruction
    user_content = (
        f"Summarize the following document in approximately {actual_target} words. "
        f"Make sure to complete all sentences properly.\n\n"
        f"DOCUMENT:\n{full_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # Use apply_chat_template if available
    try:
        prompt = model_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        log("Using chat template")
    except:
        # Fallback to manual template
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        log("Using manual template")

    log(f"Prompt length: {len(prompt)} chars")

    inputs = model_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=24576  # Qwen3 supports 32K native context
    ).to(model.device)

    log(f"Input tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=min(actual_target * 5, 4096),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            pad_token_id=model_tokenizer.pad_token_id,
            eos_token_id=model_tokenizer.eos_token_id
        )

    log(f"Output tokens: {output.shape[1]}")

    # Decode only the new tokens
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    decoded = model_tokenizer.decode(new_tokens, skip_special_tokens=True)

    log(f"Decoded summary length: {len(decoded)} chars, {len(decoded.split())} words")

    # Clean up think tags and special tokens
    decoded = decoded.strip()
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()

    # Limit to max words
    result = limit_words(decoded, actual_target)

    log(f"Final summary: {len(result)} chars, {len(result.split())} words")

    return result

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    input_data = event["input"]

    pages = input_data["pages"]
    max_words = int(input_data.get("n_words", 500))
    system_prompt = input_data.get("system_prompt", DEFAULT_SUMMARY_PROMPT)

    log(f"Processing {len(pages)} pages, target: {max_words} words")

    # Load single model
    load_model()

    # 1️⃣ Translate pages
    log("Starting translation...")
    start = time.time()
    for i, p in enumerate(pages):
        log(f"Translating page {i+1}/{len(pages)}")
        p["text"] = translate_text(p["text"])
    log(f"Translation done in {time.time()-start:.2f}s")

    # 2️⃣ Summarize
    log(f"Creating summary ({max_words} words)")
    start = time.time()
    summary = summarize_all_pages(pages, max_words, system_prompt)
    log(f"Summary done in {time.time()-start:.2f}s")

    if not summary:
        log("WARNING: Summary is empty!")

    log("Handler finished")

    return {
        "summary": summary,
        "pages": pages
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})
