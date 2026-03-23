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

# Cached prompt token IDs to avoid re-tokenizing the system prompt every call
_cached_translate_prompts = {}

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

def build_translate_prompt(target_language: str) -> str:
    """Build a translation system prompt for the given target language.
    The model will auto-detect the source language."""
    prompt = (
        f"You are a certified professional legal translator.\n"
        f"Auto-detect the language of the input text and translate it into {target_language}.\n"
        f"STRICT RULES:\n"
        f"- Translate ONLY — do NOT summarize, paraphrase, or add commentary\n"
        f"- Preserve the original meaning, tone, and structure as closely as possible\n"
        f"- Keep proper nouns, names, dates, and numbers unchanged\n"
        f"- Preserve paragraph breaks and line structure\n"
        f"- If a word or phrase is already in {target_language}, keep it as-is\n"
        f"- Output ONLY the {target_language} translation, nothing else\n"
        f"- Do NOT include any notes, explanations, or metadata about the translation\n"
        f"- Do NOT mix languages: every word in the output MUST be in {target_language} "
        f"(except proper nouns, names, and abbreviations)\n"
        f"- Do NOT use words from other variants or related languages "
        f"(e.g. if {target_language} is Spanish, do NOT use Catalan, Portuguese, or Italian words)\n"
        f"CONSISTENCY RULES (CRITICAL):\n"
        f"- Every transliterated name MUST be spelled EXACTLY the same way every time it appears\n"
        f"- Preserve ALL letters in transliterated names — do NOT drop, swap, or shorten syllables\n"
        f"- Example: if 'Пожитков' → 'Pozhitkov', it must ALWAYS be 'Pozhitkov' (NEVER 'Pogotikov', 'Pozhikov', etc.)\n"
        f"- Example: if 'Курина' → 'Kurina', it must ALWAYS be 'Kurina' (NEVER 'Kurna')\n"
        f"- Russian street names: preserve the full genitive form. Example: 'ул. Герасима Курина' → 'calle Gerasima Kurina' (NOT 'calle Gerasim Kurna')\n"
        f"- Do NOT repeat or duplicate content blocks — translate each section exactly once\n"
    )

    # Add Cyrillic-specific transliteration rules (apply when source is Russian/Cyrillic)
    prompt += (
        "CYRILLIC TRANSLITERATION RULES (apply ONLY if source text contains Cyrillic script):\n"
        "- Use phonetic transliteration that matches the Cyrillic spelling letter-by-letter\n"
        "- Complete mapping: А→A, Б→B, В→V, Г→G, Д→D, Е→E, Ё→Yo, Ж→Zh, З→Z, "
        "И→I, Й→Y, К→K, Л→L, М→M, Н→N, О→O, П→P, Р→R, С→S, Т→T, "
        "У→U, Ф→F, Х→Kh, Ц→Ts, Ч→Ch, Ш→Sh, Щ→Shch, Ъ→(omit), Ы→Y, Ь→(omit), Э→E, Ю→Yu, Я→Ya\n"
        "- 'Кс' in Russian names → 'Ks'. Example: Ксенофонтов → Ksenofontov (NOT Xenofontov)\n"
        "- BUT if a Cyrillic name is a phonetic rendering of a known foreign name, restore the original spelling. "
        "Example: КСАВЬЕР → XAVIER (NOT KSAVIER)\n"
        "- 'Е' → always 'E' (NEVER 'I'). Example: ГРИСЕН → GRISEN (NOT GRISIN)\n"
        "- 'Ж' → always 'Zh' (NEVER skip it). Example: Пожитков → Pozhitkov (NEVER Pogotikov)\n"
        "- Company names in Cyrillic are phonetic transcriptions — transliterate them back faithfully\n"
        "- For foreign place names, streets, and districts written phonetically in Cyrillic, "
        "ALWAYS restore the official English name — do NOT transliterate.\n"
        "HONG KONG: Сёнвань → Sheung Wan, Коулун → Kowloon, Цим Ша Цуй → Tsim Sha Tsui, "
        "Бонэм Стрэнд → Bonham Strand, Ванчай → Wan Chai, Монгкок → Mong Kok, "
        "Централ → Central, Абердин → Aberdeen, Чайвань → Chai Wan, Куорри Бей → Quarry Bay\n"
        "UAE: Дубай → Dubai, Абу-Даби → Abu Dhabi, Шарджа → Sharjah, "
        "Джебел Али → Jebel Ali, Дейра → Deira, Бур Дубай → Bur Dubai, "
        "Аджман → Ajman, Рас-эль-Хайма → Ras Al Khaimah, Фуджейра → Fujairah\n"
        "UK: Лондон → London, Вестминстер → Westminster, Кэнэри Уорф → Canary Wharf, "
        "Эдинбург → Edinburgh, Манчестер → Manchester, Бирмингем → Birmingham\n"
        "CYPRUS: Никосия → Nicosia, Лимассол → Limassol, Ларнака → Larnaca, Пафос → Paphos\n"
        "SINGAPORE: Сингапур → Singapore, Раффлз Плейс → Raffles Place\n"
        "BVI: Тортола → Tortola, Род Таун → Road Town\n"
        "SEYCHELLES: Маэ → Mahe, Виктория → Victoria, Праслин → Praslin\n"
        "OTHER: Панама → Panama, Белиз → Belize, Гибралтар → Gibraltar, "
        "Лихтенштейн → Liechtenstein, Люксембург → Luxembourg, Мальта → Malta, "
        "Каймановы острова → Cayman Islands, Бермуды → Bermuda\n"
        "COMMON TERMS: Стрит/Стрэнд → Street/Strand, Билдинг → Building, "
        "Башня/Тауэр → Tower, Авеню → Avenue, Плаза → Plaza, Роуд → Road\n"
    )

    # Add Russian institution/abbreviation mappings
    prompt += (
        "RUSSIAN ABBREVIATIONS AND INSTITUTIONS:\n"
        "- ОВД (Отдел Внутренних Дел) → Departamento de Policía / Police Department (NOT 'Oficina de Investigación de Delitos')\n"
        "- ЗАГС → Registro Civil / Civil Registry\n"
        "- ИНН → NIF (Número de Identificación Fiscal) / TIN (Tax Identification Number)\n"
        "- ОГРН → Número de Registro Estatal / State Registration Number\n"
    )

    # Add Spanish-specific legal terminology if target is Spanish
    if target_language.lower() in ("spanish", "español", "espanol"):
        prompt += (
            "SPANISH LEGAL TERMINOLOGY (MANDATORY — use these exact terms):\n"
            "PARTIES IN LEASE/RENTAL AGREEMENTS (CRITICAL — be consistent throughout):\n"
            "- Tenant → Arrendatario (NEVER 'Inquilino' — use 'Arrendatario' EVERYWHERE in the document)\n"
            "- Landlord (singular) → Arrendador\n"
            "- Landlords (plural) → Arrendadores\n"
            "- CRITICAL: If the document uses plural 'Landlords', ALWAYS use 'los Arrendadores' (NEVER 'el Arrendador')\n"
            "- CRITICAL: Pick ONE term for each party and use it CONSISTENTLY throughout the ENTIRE document. "
            "Do NOT alternate between 'Inquilino' and 'Arrendatario' — ALWAYS use 'Arrendatario'.\n"
            "PARTIES IN LOAN AGREEMENTS:\n"
            "- Lender / Займодавец → Prestamista (NEVER 'Cedente', NEVER 'Acreedor', NEVER 'Creditor')\n"
            "- Borrower / Заемщик → Prestatario (NEVER 'Deudor')\n"
            "- Creditor / Кредитор → Acreedor\n"
            "- Debtor / Должник → Deudor\n"
            "- Цедент → Cedente (ONLY in cession/assignment agreements)\n"
            "- Цессионарий → Cesionario (ONLY in cession/assignment agreements)\n"
            "OTHER LEGAL PARTIES:\n"
            "- party (legal) → parte (NEVER 'partido')\n"
            "- parties → partes (NEVER 'partidos')\n"
            "- trespasser → ocupante ilegal (NEVER 'intruso')\n"
            "- witnesses → testigos\n"
            "CONTRACT STRUCTURE TERMS:\n"
            "- Schedule (contract appendix) → Anexo (NEVER 'Programa')\n"
            "- Schedule A, Schedule B → Anexo A, Anexo B\n"
            "- Clause → Cláusula\n"
            "- Exhibit → Exhibición / Anexo\n"
            "- Addendum → Adenda\n"
            "- Amendment → Enmienda\n"
            "COMPANY TYPES:\n"
            "- ОАО (Открытое Акционерное Общество) → Sociedad Anónima (S.A.) — NEVER 'Societat Anónima'\n"
            "- ЗАО (Закрытое Акционерное Общество) → Sociedad Anónima Cerrada\n"
            "- ООО (Общество с Ограниченной Ответственностью) → Sociedad de Responsabilidad Limitada (S.R.L.) — NEVER 'Sociedad con Limitación'\n"
            "- Limited / Ltd → Limitada / Ltda.\n"
            "- АО (Акционерное Общество) → Sociedad Anónima (S.A.)\n"
            "- ИП (Индивидуальный Предприниматель) → Empresario Individual\n"
            "- Международная Акционерная Компания → Compañía Internacional Sociedad Anónima — NEVER 'Compañía Internacional de Acciones'\n"
            "- УК (Управляющая Компания) → Sociedad Gestora / Compañía Gestora\n"
            "REAL ESTATE AND LEASE TERMS:\n"
            "- lease → contrato de arrendamiento\n"
            "- rent → renta / alquiler\n"
            "- premises → local / instalaciones\n"
            "- nuisance → molestias / actividades molestas (NEVER leave as 'nuisance' in English)\n"
            "- shareholding / equity stake → participación accionaria / porcentaje de acciones\n"
            "- remedies → recursos legales / acciones legales (NEVER 'remedios')\n"
            "- written notice → notificación escrita\n"
            "- 'three (3) months notice' → 'con tres (3) meses de antelación' (NEVER 'tres meses antes')\n"
            "- 'it is hereby agreed as follows' → 'EN CONSECUENCIA, LAS PARTES ACUERDAN LO SIGUIENTE'\n"
            "- act of God → fuerza mayor (preferred) or acto de fuerza mayor\n"
            "ARCHITECTURAL AND BUILDING TERMS:\n"
            "- basement / underground floor → sótano (NEVER 'planta baja subterránea')\n"
            "- mezzanine / mezzanine floor → entresuelo (NEVER 'plaza media')\n"
            "- ground floor → planta baja\n"
            "- floor plan → plano de planta\n"
            "FINANCIAL AND LEGAL TERMS:\n"
            "- расчеты / settlements → pagos / liquidaciones (NEVER 'cálculo')\n"
            "- Договор займа → Contrato de Préstamo\n"
            "- Договор цессии → Contrato de Cesión\n"
            "- Устав → Estatutos Sociales\n"
            "- Доверенность → Poder Notarial\n"
            "- Протокол → Acta\n"
            "- Решение → Resolución / Decisión\n"
            "- по решению / по усмотрению → por decisión de (NEVER 'a discreción')\n"
            "- месторождения → yacimientos de recursos naturales\n"
            "- новые области природных ресурсов → nuevas áreas de recursos naturales (NOT just 'yacimientos')\n"
            "- прошито и пронумеровано → cosido y numerado (NOT 'pegado')\n"
            "- Наблюдательный совет / Технический комитет → Comité Técnico (NOT 'Comité de Supervisión' unless context is a supervisory board)\n"
            "GENERAL RULES FOR SPANISH:\n"
            "- Use standard Castilian Spanish (castellano) — NEVER Catalan, Galician, or other variants\n"
            "- Use formal legal register: use 'deberá' for obligations, 'por la presente' for declarations\n"
            "- Use standard Spanish date format: '__ de octubre de 2025'\n"
            "- Maintain formal legal phrasing: 'representado por su director', 'actuando en virtud de'\n"
            "- ALL English words MUST be translated — do NOT leave any English terms in the output "
            "(except proper nouns, company names, and internationally recognized abbreviations)\n"
            "- If Greek text appears (e.g. architectural labels), add Spanish translation in brackets: e.g. 'ΚΑΤΟΨΗ ΥΠΟΓΕΙΟΥ [PLANO DEL SÓTANO]'\n"
        )

    # Add legal translation standards (for all languages)
    prompt += (
        "LEGAL TRANSLATION STANDARDS:\n"
        f"- Use formal legal register in {target_language}\n"
        "- Preserve civil law terminology accurately\n"
        "- Maintain formal legal phrasing and tone\n"
        f"- Use standard date format for {target_language}\n"
        "- Do NOT leave any terms in English unless they are proper nouns or internationally recognized abbreviations\n"
    )

    return prompt


# =====================================================
# Load single Qwen3-14B model — auto-detects GPU caps
# =====================================================
def _detect_gpu_config():
    """Detect GPU capabilities at runtime and return optimal dtype + device info."""
    if not torch.cuda.is_available():
        log("WARNING: No CUDA GPU detected, running on CPU (very slow)")
        return torch.float32, "cpu"

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)

    log(f"GPU detected: {gpu_name}")
    log(f"  Compute capability: {capability[0]}.{capability[1]}")
    log(f"  VRAM: {vram_gb:.1f} GB")

    # BF16 support: Ampere (sm_80+) → A40, RTX A6000, A100, RTX 3090, etc.
    # FP16 fallback: Turing (sm_75) and older → V100, T4, RTX 2080, etc.
    if capability[0] >= 8:
        dtype = torch.bfloat16
        log(f"  Using BF16 (native support on {gpu_name})")
    else:
        dtype = torch.float16
        log(f"  Using FP16 (BF16 not supported on {gpu_name})")

    return dtype, gpu_name


def load_model():
    global model_tokenizer, model
    if model is not None:
        return

    dtype, gpu_name = _detect_gpu_config()

    log(f"Loading OpenPipe/Qwen3-14B-Instruct ({dtype})")

    model_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    # Try to load with Flash Attention 2 for major speedup
    # Works on both A40 and RTX A6000 (both Ampere, sm_86)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        log("Loaded with Flash Attention 2")
    except Exception as e:
        log(f"Flash Attention 2 not available ({e}), falling back to eager attention")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

    model.eval()

    # Enable CUDA optimizations (TF32 supported on Ampere+)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Try torch.compile for additional speedup (PyTorch 2.x)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        log("torch.compile applied (reduce-overhead mode)")
    except Exception as e:
        log(f"torch.compile not available: {e}")

    log(f"Model loaded on {gpu_name} | dtype={dtype} | "
        f"device={model.device if hasattr(model, 'device') else 'multi-gpu'}")

# =====================================================
# Detect layout separators
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

# =====================================================
# Build and cache the system prompt token IDs
# =====================================================
def get_translate_prefix_ids(target_language: str):
    """Build the tokenized system prompt prefix once, and cache it."""
    if target_language in _cached_translate_prompts:
        return _cached_translate_prompts[target_language]

    translate_prompt = build_translate_prompt(target_language)

    # Build the template with a placeholder for user content
    messages = [
        {"role": "system", "content": translate_prompt},
    ]

    try:
        prefix = model_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except:
        prefix = f"<|im_start|>system\n{translate_prompt}<|im_end|>\n"

    prefix_ids = model_tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
    _cached_translate_prompts[target_language] = (translate_prompt, prefix_ids)
    log(f"Cached translate prompt for '{target_language}': {prefix_ids.shape[1]} tokens")
    return translate_prompt, prefix_ids


# =====================================================
# TRANSLATION — Optimized batch processing
# =====================================================
def translate_text_batch(texts: list, target_language: str = "English") -> list:
    """Translate multiple texts (pages) efficiently by merging small pages
    and minimizing inference calls."""
    if not texts:
        return texts

    # Get cached system prompt
    translate_prompt, _ = get_translate_prefix_ids(target_language)

    # ---- Step 1: Merge small pages into larger chunks ----
    # This reduces the number of inference calls dramatically
    MERGE_WORD_LIMIT = 2500  # Qwen3 handles 32K context; we can go bigger
    merged_chunks = []
    current_pages = []
    current_word_count = 0

    for idx, text in enumerate(texts):
        if not text or not text.strip():
            # Track empty pages so we can map results back
            if current_pages:
                merged_chunks.append(current_pages)
                current_pages = []
                current_word_count = 0
            merged_chunks.append([(idx, text, True)])  # True = skip (empty)
            continue

        word_count = len(text.split())

        # If adding this page would exceed limit, flush current
        if current_word_count + word_count > MERGE_WORD_LIMIT and current_pages:
            merged_chunks.append(current_pages)
            current_pages = []
            current_word_count = 0

        current_pages.append((idx, text, False))
        current_word_count += word_count

    if current_pages:
        merged_chunks.append(current_pages)

    log(f"Merged {len(texts)} pages into {len(merged_chunks)} inference calls")

    # ---- Step 2: Translate each merged chunk ----
    results = [""] * len(texts)

    for chunk_idx, page_group in enumerate(merged_chunks):
        # Check if this is a skip group
        if len(page_group) == 1 and page_group[0][2]:
            idx, text, _ = page_group[0]
            results[idx] = text
            continue

        # Combine pages with clear separators
        combined_parts = []
        for idx, text, _ in page_group:
            stripped = text.strip()
            # Skip chunks with too few letter characters
            if len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 5:
                results[idx] = text
                continue
            combined_parts.append((idx, stripped))

        if not combined_parts:
            continue

        # If only one page, translate directly (no separators needed)
        if len(combined_parts) == 1:
            idx, stripped = combined_parts[0]
            user_text = stripped
        else:
            # Use clear page separators
            parts = []
            for i, (idx, stripped) in enumerate(combined_parts):
                parts.append(f"=== PAGE {i+1} ===\n{stripped}")
            user_text = "\n\n".join(parts)

        word_count = len(user_text.split())
        log(f"Translating chunk {chunk_idx+1}/{len(merged_chunks)} "
            f"({len(combined_parts)} pages, {word_count} words)")

        messages = [
            {"role": "system", "content": translate_prompt},
            {"role": "user", "content": user_text}
        ]

        try:
            prompt = model_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = (
                f"<|im_start|>system\n{translate_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = model_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=24576
        ).to(model.device)

        input_len = inputs['input_ids'].shape[1]
        log(f"  Input tokens: {input_len}")

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(word_count * 3, 8192),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=model_tokenizer.pad_token_id,
                eos_token_id=model_tokenizer.eos_token_id
            )

        new_tokens = output[0][input_len:]
        decoded = model_tokenizer.decode(new_tokens, skip_special_tokens=True)
        gen_time = time.time() - t0
        gen_tokens = len(new_tokens)
        log(f"  Generated {gen_tokens} tokens in {gen_time:.1f}s "
            f"({gen_tokens/gen_time:.1f} tok/s)")

        # Clean up think tags and special tokens
        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
        decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()

        # Remove echoed system prompt if model regurgitated instructions
        for marker in ["STRICT RULES:", "LEGAL TRANSLATION STANDARDS:"]:
            if marker in decoded:
                idx_m = decoded.find(marker)
                after = decoded[idx_m:]
                lines = after.split("\n")
                last_rule = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("- "):
                        last_rule = i
                decoded = "\n".join(lines[last_rule + 1:]).strip()

        # Map results back to individual pages
        if len(combined_parts) == 1:
            idx, _ = combined_parts[0]
            results[idx] = decoded
        else:
            # Split by page separators
            page_translations = re.split(r"===\s*PAGE\s+\d+\s*===", decoded)
            # Remove empty first element if present
            page_translations = [p.strip() for p in page_translations if p.strip()]

            for i, (idx, _) in enumerate(combined_parts):
                if i < len(page_translations):
                    results[idx] = page_translations[i]
                else:
                    # Fallback: if separator splitting failed, give remaining text
                    # to the last page
                    log(f"  WARNING: Could not split page {i+1}, using full output")
                    results[idx] = decoded

    return results


def translate_text(text: str, target_language: str = "English") -> str:
    """Auto-detect source language and translate to target_language using Qwen 14B.
    Single-text wrapper around the batch function."""
    result = translate_text_batch([text], target_language)
    return result[0]


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
    target_language = input_data.get("target_language", "English")

    log(f"Processing {len(pages)} pages, target: {max_words} words, translate to: {target_language}")

    # Load single model
    load_model()

    # 1️⃣ Translate pages — BATCH (all pages at once, merged into fewer calls)
    log(f"Starting batch translation to {target_language}...")
    start = time.time()
    page_texts = [p["text"] for p in pages]
    translated_texts = translate_text_batch(page_texts, target_language)
    for i, p in enumerate(pages):
        p["text"] = translated_texts[i]
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
