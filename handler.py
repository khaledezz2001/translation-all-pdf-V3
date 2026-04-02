import time
import re
import runpod
from vllm import LLM, SamplingParams

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

MODEL_PATH = "/models/hf/qwen"
llm_engine = None
tokenizer = None

# =====================================================
# System prompts (unchanged)
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

    if target_language.lower() in ("spanish", "español", "espanol"):
        prompt += (
            "RUSSIAN ABBREVIATIONS AND INSTITUTIONS:\n"
            "- ОВД (Отдел Внутренних Дел) → Departamento de Policía (NOT 'Oficina de Investigación de Delitos')\n"
            "- ЗАГС → Registro Civil\n"
            "- ИНН → NIF (Número de Identificación Fiscal)\n"
            "- ОГРН → Número de Registro Estatal\n"
        )
    elif target_language.lower() in ("french", "français", "francais"):
        prompt += (
            "RUSSIAN ABBREVIATIONS AND INSTITUTIONS:\n"
            "- ОВД (Отдел Внутренних Дел) → Département de Police / Service des Affaires Intérieures\n"
            "- ЗАГС → Bureau de l'État Civil\n"
            "- ИНН → Numéro d'Identification Fiscale (NIF)\n"
            "- ОГРН → Numéro d'Enregistrement National\n"
        )
    elif target_language.lower() in ("german", "deutsch"):
        prompt += (
            "RUSSIAN ABBREVIATIONS AND INSTITUTIONS:\n"
            "- ОВД (Отдел Внутренних Дел) → Polizeidienststelle\n"
            "- ЗАГС → Standesamt\n"
            "- ИНН → Steueridentifikationsnummer (Steuer-IdNr.)\n"
            "- ОГРН → Staatliche Registrierungsnummer\n"
        )
    else:
        prompt += (
            "RUSSIAN ABBREVIATIONS AND INSTITUTIONS:\n"
            "- ОВД (Отдел Внутренних Дел) → Police Department\n"
            "- ЗАГС → Civil Registry\n"
            "- ИНН → TIN (Tax Identification Number)\n"
            "- ОГРН → State Registration Number\n"
        )

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

    # Add Arabic-specific legal terminology if target is Arabic
    if target_language.lower() in ("arabic", "العربية", "عربي"):
        prompt += (
            "ARABIC LEGAL TERMINOLOGY (MANDATORY — use these exact terms):\n"
            "CRITICAL SCRIPT RULES:\n"
            "- The ENTIRE output MUST be in Arabic script — NO Cyrillic, NO Korean, NO Chinese, NO Latin characters "
            "(except proper nouns like company registration numbers, addresses in Latin script, and internationally recognized abbreviations)\n"
            "- ALL Russian words MUST be translated or transliterated into Arabic script — NEVER leave Cyrillic text\n"
            "- ALL numbers written in words MUST be translated into Arabic words. "
            "Example: 'сто пятьдесят семь миллионов пятьсот тысяч' → 'مائة وسبعة وخمسون مليوناً وخمسمائة ألف' (NEVER leave Russian number words)\n"
            "- 'сто восемьдесят' → 'مائة وثمانون' (NEVER leave in Russian)\n"
            "CYRILLIC TO ARABIC TRANSLITERATION (for names and places):\n"
            "- А→ا, Б→ب, В→ف, Г→غ, Д→د, Е→ي/إ, Ё→يو, Ж→ج, З→ز, "
            "И→ي, Й→ي, К→ك, Л→ل, М→م, Н→ن, О→و, П→ب, Р→ر, С→س, Т→ت, "
            "У→و, Ф→ف, Х→خ, Ц→تس, Ч→تش, Ш→ش, Щ→شتش, Э→إ, Ю→يو, Я→يا\n"
            "- ТЕХНОЛОДЖИ → تكنولوجي (NEVER use Korean, Chinese, or other non-Arabic characters)\n"
            "- Every Cyrillic name MUST be fully transliterated into Arabic letters\n"
            "COMPANY TYPES:\n"
            "- ООО (Общество с Ограниченной Ответственностью) → شركة ذات مسؤولية محدودة\n"
            "- ОАО (Открытое Акционерное Общество) → شركة مساهمة عامة\n"
            "- ЗАО (Закрытое Акционерное Общество) → شركة مساهمة مقفلة\n"
            "- АО (Акционерное Общество) → شركة مساهمة\n"
            "- ИП (Индивидуальный Предприниматель) → مؤسسة فردية\n"
            "- ПАО (Публичное Акционерное Общество) → شركة مساهمة عامة\n"
            "- НПО (Научно-Производственное Объединение) → الاتحاد العلمي الإنتاجي\n"
            "- УК (Управляющая Компания) → شركة إدارية\n"
            "- GMBH / GmbH → ش.ذ.م.م (شركة ذات مسؤولية محدودة)\n"
            "- Limited / Ltd → المحدودة\n"
            "MEETING AND PROTOCOL TERMS:\n"
            "- Протокол → محضر\n"
            "- Общее собрание участников → الجمعية العامة للشركاء\n"
            "- Внеочередное общее собрание → الجمعية العامة غير العادية\n"
            "- Повестка дня → جدول الأعمال\n"
            "- Слушали → استُمع إلى\n"
            "- Голосовали → التصويت\n"
            "- Решили → تقرر\n"
            "- За → موافق\n"
            "- Против → معارض\n"
            "- Воздержался → ممتنع\n"
            "- Председатель собрания → رئيس الجلسة\n"
            "- Секретарь собрания → أمين سر الجلسة\n"
            "- Ревизионная комиссия → لجنة المراجعة\n"
            "- Крупная сделка → صفقة كبرى\n"
            "LEGAL AND FINANCIAL TERMS:\n"
            "- Устав → النظام الأساسي\n"
            "- Доверенность → توكيل / وكالة\n"
            "- Договор поставки → عقد توريد\n"
            "- Договор займа → عقد قرض\n"
            "- Договор цессии → عقد حوالة حق\n"
            "- Решение → قرار\n"
            "- расчеты / settlements → مدفوعات / تسويات\n"
            "- ОГРН → رقم التسجيل الحكومي الرئيسي\n"
            "- ИНН → رقم التعريف الضريبي\n"
            "- КПП → رمز سبب التسجيل\n"
            "- БИК → رمز التعريف المصرفي\n"
            "- Расчетный счет → الحساب الجاري\n"
            "- Корреспондентский счет → حساب المراسلة\n"
            "- НДС → ضريبة القيمة المضافة\n"
            "- уставный капитал → رأس المال التأسيسي\n"
            "- балансовая стоимость → القيمة الدفترية\n"
            "PARTIES IN CONTRACTS:\n"
            "- Покупатель → المشتري\n"
            "- Поставщик → المورّد\n"
            "- Арендатор → المستأجر\n"
            "- Арендодатель → المؤجّر\n"
            "- Займодавец → المُقرض\n"
            "- Заемщик → المقترض\n"
            "- Цедент → المحيل\n"
            "- Цессионарий → المحال إليه\n"
            "- party (legal) → طرف\n"
            "- parties → أطراف\n"
            "CONTRACT STRUCTURE TERMS:\n"
            "- Приложение → ملحق\n"
            "- Спецификация → المواصفات\n"
            "- Техническое задание → المهمة الفنية\n"
            "- Пусконаладочные работы → أعمال التشغيل والتجربة\n"
            "- Акт → محضر / سند\n"
            "RUSSIAN INSTITUTIONS:\n"
            "- ОВД (Отдел Внутренних Дел) → إدارة الشؤون الداخلية\n"
            "- УВД → مديرية الشؤون الداخلية\n"
            "- ЗАГС → مكتب السجل المدني\n"
            "- Нотариус → كاتب العدل\n"
            "GENERAL RULES FOR ARABIC:\n"
            "- Use Modern Standard Arabic (الفصحى) — NEVER colloquial dialects\n"
            "- Use formal legal register throughout\n"
            "- Use Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) OR Western Arabic numerals (0123456789) — be consistent\n"
            "- Use standard Arabic date format: '15 مارس 2024'\n"
            "- ALL text in the output MUST be readable in Arabic — absolutely NO Cyrillic, Korean, or other foreign script\n"
            "- If a Russian company or institution name appears, transliterate it FULLY into Arabic script\n"
            "- For well-known foreign company names (e.g. ALFA-BANK), use the accepted Arabic name: ألفا بنك\n"
        )

    # Add French-specific legal terminology if target is French
    if target_language.lower() in ("french", "français", "francais"):
        prompt += (
            "FRENCH LEGAL TERMINOLOGY (MANDATORY — use these exact terms):\n"
            "CONSISTENCY AND QUALITY RULES (HIGHEST PRIORITY):\n"
            "- COMPANY NAME CONSISTENCY: If a company name appears in the source (in ANY script — Latin, Cyrillic, etc.), "
            "you MUST use ONE single spelling for that company throughout the ENTIRE French output. "
            "Example: if the English part uses 'DARBSHIRE LTD', the Russian part's 'ДАРБШИР ЛИМИТЕД' must ALSO be rendered as 'DARBSHIRE LTD' in French — "
            "NEVER 'DARBEESHIR', 'Darbishire', 'DARBASHIR', 'Darbyshire', or any other variant. "
            "ALWAYS prefer the Latin-script spelling from the original if one exists.\n"
            "- DEFINED TERM CONSISTENCY: When the source document defines a term (e.g. 'the Shop', 'the Tenant', 'the Landlords'), "
            "use ONE French equivalent for that defined term throughout. "
            "Example: 'Shop' → 'Magasin' EVERYWHERE — NEVER alternate between 'Magasin' and 'local' within the same document.\n"
            "- NUMBERING CORRECTION: If the source document contains numbering errors (e.g. duplicate clause numbers), "
            "CORRECT them in the French translation. Do NOT reproduce numbering mistakes from the source. "
            "Renumber sequentially so every clause has a unique number.\n"
            "- NO LITERAL TRANSLATION OF AWKWARD SOURCE: If the source text is poorly drafted or contains awkward phrasing, "
            "translate the INTENDED MEANING in natural, correct French rather than producing a word-for-word rendering. "
            "Example: 'with whom the Landlords have been charged' → 'dont les Bailleurs ont été tenus responsables' "
            "(NOT 'avec laquelle les Bailleurs ont été chargés').\n"
            "- ALL ENGLISH WORDS MUST BE TRANSLATED: Do NOT leave ANY English word untranslated in the French output "
            "(except proper nouns, company names, and internationally recognized abbreviations). "
            "Common mistakes to avoid: 'advisers' must be 'conseillers' (NEVER leave 'advisers' in English), "
            "'consultants' → 'consultants' (same in French), 'nuisance' → 'nuisance' or 'trouble' (French word exists).\n"
            "- NO SECTION DUPLICATION: Translate each section of the source document exactly ONCE. "
            "NEVER repeat or duplicate paragraphs, clauses, or sections. If the source has sections 5, 6, 7, "
            "the French must have exactly one section 5, one section 6, one section 7.\n"
            "PARTIES IN LEASE/RENTAL AGREEMENTS:\n"
            "- Tenant → Locataire (NEVER 'Preneur' unless explicitly a commercial lease)\n"
            "- Landlord → Bailleur\n"
            "- Landlords (plural) → Bailleurs\n"
            "- Shop (as a defined term in lease) → Magasin (ALWAYS 'Magasin', NEVER 'local' when referring to the defined 'Shop')\n"
            "- CRITICAL: Pick ONE term for each party and use it CONSISTENTLY throughout the ENTIRE document.\n"
            "PARTIES IN LOAN AGREEMENTS:\n"
            "- Lender / Займодавец → Prêteur\n"
            "- Borrower / Заемщик → Emprunteur\n"
            "- Creditor / Кредитор → Créancier\n"
            "- Debtor / Должник → Débiteur\n"
            "- Цедент → Cédant (ONLY in cession/assignment agreements)\n"
            "- Цессионарий → Cessionnaire (ONLY in cession/assignment agreements)\n"
            "OTHER LEGAL PARTIES:\n"
            "- party (legal) → partie (NEVER 'parti')\n"
            "- parties → parties\n"
            "- trespasser → occupant illégal\n"
            "- witnesses → témoins\n"
            "CONTRACT STRUCTURE TERMS:\n"
            "- Schedule (contract appendix) → Annexe (NEVER 'Programme') — NOTE: 'Annexe' is FEMININE in French: "
            "'la première annexe', 'de la première annexe' (NEVER 'du premier annexe')\n"
            "- Schedule A, Schedule B → Annexe A, Annexe B\n"
            "- Clause → Clause / Article\n"
            "- Exhibit → Pièce jointe / Annexe\n"
            "- Addendum → Avenant\n"
            "- Amendment / amending → Modification / Avenant; the verb 'to amend' → 'modifier' or 'amender' "
            "(NEVER 'amener' — 'amener' means 'to bring', which is WRONG)\n"
            "COMPANY TYPES (CRITICAL — match the legal form precisely):\n"
            "- ОАО (Открытое Акционерное Общество) → Société Anonyme Ouverte\n"
            "- ЗАО (Закрытое Акционерное Общество) → Société Anonyme Fermée\n"
            "- ООО (Общество с Ограниченной Ответственностью) → Société à Responsabilité Limitée (SARL) — "
            "This is an LLC, NEVER translate as 'Société Anonyme (SA)'\n"
            "- Limited / Ltd → Limitée / Ltée\n"
            "- АО (Акционерное Общество) → Société Anonyme (SA)\n"
            "- ИП (Индивидуальный Предприниматель) → Entrepreneur Individuel\n"
            "- УК (Управляющая Компания) → Société de Gestion\n"
            "- МКАО (Международная Компания Акционерное Общество) → Société Internationale Anonyme — "
            "Do NOT leave 'MKAO' as is. ALWAYS expand Russian abbreviations into their French equivalent. "
            "Avoid redundant phrasing like 'Société Internationale de la Société Anonyme' — use 'Société Internationale Anonyme'.\n"
            "COMMITTEES AND BODIES:\n"
            "- Технический комитет / Technical Committee → Comité technique (CT) — "
            "NEVER translate as 'Comité de surveillance' (that means 'Supervisory Committee', which is different). "
            "Use 'CT' as the abbreviation CONSISTENTLY — NEVER switch to 'TK' (Russian transliteration).\n"
            "- Наблюдательный совет / Supervisory Board → Conseil de surveillance (ONLY if the source says supervisory/наблюдательный)\n"
            "REAL ESTATE AND LEASE TERMS:\n"
            "- lease → bail / contrat de location\n"
            "- rent → loyer\n"
            "- premises → locaux / lieux loués\n"
            "- nuisance → nuisance / trouble de voisinage\n"
            "- shareholding / equity stake → participation au capital\n"
            "- remedies → recours / voies de recours (NEVER 'remèdes')\n"
            "- written notice → notification écrite / mise en demeure\n"
            "- 'three (3) months notice' → 'un préavis de trois (3) mois'\n"
            "- 'it is hereby agreed as follows' → 'IL EST CONVENU CE QUI SUIT'\n"
            "- act of God → force majeure / cas de force majeure\n"
            "- security deposit / such amount from the Tenant → dépôt de garantie / cette somme reçue du Locataire "
            "(NEVER 'cette somme du Locataire' which is ambiguous)\n"
            "ARCHITECTURAL AND BUILDING TERMS:\n"
            "- basement / underground floor → sous-sol\n"
            "- mezzanine / mezzanine floor → mezzanine / entresol\n"
            "- ground floor → rez-de-chaussée\n"
            "- floor plan → plan d'étage\n"
            "FINANCIAL AND LEGAL TERMS:\n"
            "- расчеты / settlements → paiements / règlements\n"
            "- financial assistance → assistance financière (NEVER 'secours financier' — that means emergency relief)\n"
            "- Договор займа → Contrat de Prêt\n"
            "- Договор цессии → Contrat de Cession\n"
            "- Устав → Statuts\n"
            "- Доверенность → Procuration\n"
            "- Протокол → Procès-verbal\n"
            "- Решение → Décision / Résolution\n"
            "- по решению / по усмотрению → à la discrétion de / par décision de\n"
            "- прошито и пронумеровано → relié et numéroté\n"
            "- прошито, пронумеровано, опечатано → relié, numéroté et revêtu du sceau officiel\n"
            "- Authority / competent authority → Autorité compétente (NEVER 'Authorité' — that is a typo/anglicism)\n"
            "- 'any law substituting or amending' → 'toute loi se substituant ou modifiant' "
            "(NEVER 'l'amenant' — 'amener' means 'to bring', NOT 'to amend')\n"
            "RUSSIAN ABBREVIATIONS AND ACRONYMS:\n"
            "- When a Russian abbreviation appears, ALWAYS provide the full French expansion, optionally with the original in parentheses.\n"
            "- ОГРН → Numéro d'Enregistrement National (ОГРН)\n"
            "- ИНН → Numéro d'Identification Fiscale (NIF)\n"
            "- КПП → Code de Motif d'Enregistrement\n"
            "- МКАО → Société Internationale Anonyme (do NOT leave as 'MKAO')\n"
            "RUSSIAN INSTITUTIONS:\n"
            "- ОВД (Отдел Внутренних Дел) → Département de Police / Service des Affaires Intérieures\n"
            "- ЗАГС → Bureau de l'État Civil\n"
            "TRANSLITERATION CONSISTENCY RULES:\n"
            "- Patronymics must be spelled consistently: if you write 'Igorievich' once, use 'Igorievich' EVERY time "
            "(NEVER switch to 'Igorevich' or other variants within the same document).\n"
            "- Street names from the source must be reproduced EXACTLY as they appear in the original Latin-script version "
            "(if available). Example: if English says 'Mome Kapora 12', French must also say 'Mome Kapora 12' "
            "(NEVER 'Momos Kapor' or other variants).\n"
            "FRENCH GRAMMAR AND STYLE RULES:\n"
            "- Use standard Metropolitan French — NEVER Canadian French, Belgian French, or Swiss French variants\n"
            "- Use formal legal register: use 'devra' for obligations, 'par les présentes' for declarations\n"
            "- Use standard French date format: 'le 15 octobre 2025'\n"
            "- Maintain formal legal phrasing: 'représenté par son directeur', 'agissant en vertu de'\n"
            "- Use proper French accents: é, è, ê, ë, à, â, ç, ù, ô, î, ï, etc.\n"
            "- GENDER AGREEMENT: Pay careful attention to noun genders. Key examples:\n"
            "  - 'Annexe' is FEMININE: 'la première annexe' (NEVER 'le premier annexe')\n"
            "  - 'Clause' is FEMININE: 'la présente clause'\n"
            "  - 'Article' is MASCULINE: 'le présent article'\n"
            "- LEGAL VERB TENSES: For obligations and prohibitions, prefer present indicative or subjunctive "
            "over future tense. Example: 'they shall not approve' → 'ils n'approuvent pas' or 'qu'ils n'approuvent pas' "
            "(PREFERRED over 'qu'ils n'approuveront pas').\n"
            "- AVOID REDUNDANT PHRASING: Do not produce constructions like 'la Société Internationale de la Société Anonyme'. "
            "Simplify to 'la Société Internationale Anonyme'.\n"
            "- When the source text is poorly drafted (awkward, ambiguous, or ungrammatical), "
            "translate the INTENDED MEANING into natural, correct French. Do NOT produce a word-for-word calque "
            "that reproduces the source's drafting problems.\n"
        )

    # Add German-specific legal terminology if target is German
    if target_language.lower() in ("german", "deutsch"):
        prompt += (
            "GERMAN LEGAL TERMINOLOGY (MANDATORY — use these exact terms):\n"
            "CONSISTENCY AND QUALITY RULES (HIGHEST PRIORITY):\n"
            "- COMPANY NAME CONSISTENCY: If a company name appears in the source (in ANY script — Latin, Cyrillic, etc.), "
            "you MUST use ONE single spelling for that company throughout the ENTIRE German output. "
            "Example: if the English part uses 'DARBSHIRE LTD', the Russian part's 'ДАРБШИР ЛИМИТЕД' must ALSO be rendered as 'DARBSHIRE LTD' in German — "
            "NEVER 'DARBEESHIR', 'DARBEESHIRE', 'Darbishire', 'DARBASHIR', or any other variant. "
            "ALWAYS prefer the Latin-script spelling from the original if one exists.\n"
            "- DEFINED TERM CONSISTENCY: When the source document defines a term (e.g. 'the Company', 'the Director', 'the Member'), "
            "use ONE German equivalent for that defined term throughout. "
            "NEVER alternate between synonyms within the same document.\n"
            "- NO LITERAL CALQUES: Do NOT produce word-for-word translations that sound unnatural in German. "
            "Translate the INTENDED MEANING in correct, natural German legal language. "
            "Example: 'Written Decision Document' → 'Schriftlicher Beschluss' (NEVER 'Schreibendes Entscheidungsdokument'). "
            "Example: 'Memorandum of Association' → 'Gründungsurkunde' (NEVER 'Gemeinschaftsgeschäftsvertrag').\n"
            "- NO SECTION DUPLICATION: Translate each section of the source document exactly ONCE. "
            "NEVER repeat or duplicate paragraphs, clauses, or sections.\n"
            "- NUMBERING CORRECTION: If the source document contains numbering errors (e.g. duplicate clause numbers), "
            "CORRECT them in the German translation. Renumber sequentially so every clause has a unique number.\n"
            "- COMPLETE SENTENCES: NEVER leave sentences incomplete or cut off. "
            "If the source has 'and to exercise and enforce…', the German MUST complete the full clause.\n"
            "- PROPER GERMAN GRAMMAR: Pay strict attention to German grammatical cases (Nominativ, Genitiv, Dativ, Akkusativ), "
            "verb placement, reflexive pronouns, and separable verbs. "
            "Example: 'sich versichern' requires the reflexive 'sich' — NEVER omit it. "
            "Example: 'des russischen Staatsbürgers' (genitive) — NEVER 'des russischen Staatsbürger'. "
            "Example: 'das Vorhandensein des Quorums' (neuter, genitive) — NEVER 'den Vorhandensein der Quorum'.\n"
            "- CAPITALISATION: In German, ALL nouns are capitalised. Maintain consistent capitalisation throughout — "
            "NEVER write 'GESellschaft' or 'öffentlichkeit'. Correct: 'Gesellschaft', 'Öffentlichkeit'.\n"
            "- COMMA RULES: Do NOT place a comma between the subject and its conjugated verb. "
            "Example: 'Der Vorsitzende des TK berücksichtigt die Anweisungen' (NEVER 'Der Vorsitzende des TK, berücksichtigt die Anweisungen').\n"

            "CYPRUS COMPANY LAW TERMINOLOGY (CRITICAL — for English-source documents):\n"
            "- Companies Law (Cap. 113) → Gesellschaftsgesetz (Kap. 113) (NEVER 'Geschäftsgesetz')\n"
            "- Company limited by shares → Kapitalgesellschaft mit Aktienkapital (NEVER 'Geschäftsgesellschaft mit eingeschränkter Haftung')\n"
            "- Memorandum of Association → Gründungsurkunde (NEVER 'Gemeinschaftsgeschäftsvertrag', NEVER just 'Memorandum')\n"
            "- Articles of Association → Satzung (ALWAYS 'Satzung' — NEVER 'Geschäftsordnung', NEVER 'Statuten', be CONSISTENT)\n"
            "- Registered office → eingetragener Sitz / Sitz der Gesellschaft\n"
            "- Subscriber → Gründungsgesellschafter / Zeichner\n"
            "- Objects clause → Gegenstandsklausel / Unternehmensgegenstand\n"
            "- Authorised share capital → genehmigtes Aktienkapital\n"
            "- Issued share capital → ausgegebenes Aktienkapital\n"
            "- Nominal value / par value → Nennwert\n"

            "CORPORATE ROLES AND TITLES (CRITICAL — be CONSISTENT, pick ONE term per role):\n"
            "- Director (of a Ltd/Cyprus company) → Direktor (use CONSISTENTLY — NEVER alternate with 'Geschäftsführer', "
            "'Verwaltungsratsmitglied', or 'Vorstandsassessor' within the same document)\n"
            "- Board of Directors → Vorstand / Direktorium (use ONE term consistently)\n"
            "- Secretary → Sekretär / Schriftführer (use ONE term consistently)\n"
            "- Managing Director → Geschäftsführender Direktor\n"
            "- Chairman → Vorsitzender\n"
            "- For Permanent Secretary → Für den Ständigen Sekretär\n"
            "- Генеральный директор (Russian General Director) → For an Aktiengesellschaft (AG): Vorstandsvorsitzender or Vorstand. "
            "For a GmbH: Geschäftsführer. Choose based on company type and be CONSISTENT.\n"
            "- Auditor / Revisor → Wirtschaftsprüfer\n"

            "SHARE AND MEMBERSHIP TERMS (CRITICAL — be CONSISTENT):\n"
            "- Share (in a company limited by shares) → Aktie (ALWAYS 'Aktie' — NEVER alternate with 'Anteil' or 'Anteilschein')\n"
            "- Shareholder → Aktionär (ALWAYS 'Aktionär' — NEVER alternate with 'Gesellschafter' or 'Mitglied' for shareholding members)\n"
            "- Member (of a company) → Mitglied (ONLY when referring to non-shareholding membership; "
            "for shareholding members use 'Aktionär')\n"
            "- Share certificate → Aktienurkunde\n"
            "- Share transfer → Aktienübertragung\n"
            "- Ordinary share → Stammaktie\n"
            "- Preference share → Vorzugsaktie\n"
            "- Dividend → Dividende\n"
            "- Allotment → Zuteilung\n"

            "SPECIFIC COMPANY LAW PROCEDURES:\n"
            "- Pre-emption rights → Vorkaufsrechte (NEVER just 'Befugnis')\n"
            "- 'cancel the authority to sell' → 'die Vollmacht zum Verkauf widerrufen' (NEVER 'die Befugnis … zu widerrufen')\n"
            "- Lien (on shares) → Pfandrecht (an Aktien) (use CONSISTENTLY — NEVER 'Recht auf Aussetzung'; "
            "'Aussetzung' means 'suspension', which is WRONG)\n"
            "- Forfeiture of shares → Verwirkung von Aktien / für verfallen erklären "
            "(NEVER 'Konfiskation' — 'konfiszieren' is used by government authorities, NOT companies)\n"
            "- Winding up → Liquidation / Abwicklung (NEVER just 'Auflösung' — use 'in der Liquidation' for 'in winding up')\n"
            "- Extraordinary resolution → Außerordentlicher Beschluss\n"
            "- Special resolution → Sonderbeschluss\n"
            "- Ordinary resolution → Ordentlicher Beschluss\n"
            "- Annual general meeting → Ordentliche Hauptversammlung / Jahreshauptversammlung\n"
            "- Extraordinary general meeting → Außerordentliche Hauptversammlung\n"
            "- Proxy → Stimmrechtsvollmacht / Bevollmächtigter\n"
            "- Quorum → Beschlussfähigkeit / Quorum\n"
            "- Poll → Abstimmung\n"
            "- Apostille → Apostille\n"

            "RUSSIAN DOCUMENT TYPES (CRITICAL — correct German equivalents):\n"
            "- Решение единственного акционера / Written decision of sole shareholder → Schriftlicher Beschluss des alleinigen Aktionärs "
            "(NEVER 'Schreibendes Entscheidungsdokument' — that is a nonsensical calque)\n"
            "- проект договора / Draft agreement → Entwurf eines Vertrags / Vertragsentwurf "
            "(NEVER 'Vorschlag für ein Vertrag' — 'Vorschlag' means 'proposal', not 'draft')\n"
            "- Положение (internal regulation/rules) → Ordnung / Regelung / Geschäftsordnung "
            "(NEVER 'Verordnung' — 'Verordnung' means government ordinance/decree, which is TOO STRONG for internal rules)\n"
            "- Протокол заседания → Sitzungsprotokoll / Niederschrift der Sitzung\n"
            "- Устав → Satzung (for a company) / Gesellschaftsvertrag (for a GmbH)\n"

            "PARTIES IN LEASE/RENTAL AGREEMENTS:\n"
            "- Tenant → Mieter\n"
            "- Landlord → Vermieter\n"
            "- Landlords (plural) → Vermieter (plural context)\n"
            "- CRITICAL: Pick ONE term for each party and use it CONSISTENTLY throughout the ENTIRE document.\n"
            "PARTIES IN LOAN AGREEMENTS:\n"
            "- Lender / Займодавец → Darlehensgeber\n"
            "- Borrower / Заемщик → Darlehensnehmer\n"
            "- Creditor / Кредитор → Gläubiger\n"
            "- Debtor / Должник → Schuldner\n"
            "- Цедент → Zedent (ONLY in cession/assignment agreements)\n"
            "- Цессионарий → Zessionar (ONLY in cession/assignment agreements)\n"
            "OTHER LEGAL PARTIES:\n"
            "- party (legal) → Partei / Vertragspartei (NEVER 'Partei' in political sense)\n"
            "- parties → Parteien / Vertragsparteien\n"
            "- trespasser → unbefugter Betreter\n"
            "- witnesses → Zeugen\n"
            "CONTRACT STRUCTURE TERMS:\n"
            "- Schedule (contract appendix) → Anlage (NEVER 'Programm' or 'Zeitplan')\n"
            "- Schedule A, Schedule B → Anlage A, Anlage B\n"
            "- Clause → Klausel / Paragraph (§)\n"
            "- Exhibit → Anlage / Beilage\n"
            "- Addendum → Nachtrag / Ergänzung\n"
            "- Amendment → Änderung / Nachtrag\n"
            "- Appendix / Anhang → Anhang / Anlage\n"
            "- Stamp duty mark → Stempelmarke (NEVER 'Stempelzoll')\n"

            "COMPANY TYPES:\n"
            "- ОАО (Открытое Акционерное Общество) → Offene Aktiengesellschaft\n"
            "- ЗАО (Закрытое Акционерное Общество) → Geschlossene Aktiengesellschaft\n"
            "- ООО (Общество с Ограниченной Ответственностью) → Gesellschaft mit beschränkter Haftung (GmbH)\n"
            "- Limited / Ltd (when it's an LTD company) → Gesellschaft mit beschränkter Haftung (GmbH) "
            "OR keep as 'Limited' / 'Ltd' if it is a foreign company name\n"
            "- Company limited by shares (Cyprus) → Kapitalgesellschaft mit Aktienkapital "
            "(NEVER 'Geschäftsgesellschaft mit eingeschränkter Haftung')\n"
            "- АО (Акционерное Общество) → Aktiengesellschaft (AG)\n"
            "- ИП (Индивидуальный Предприниматель) → Einzelunternehmer\n"
            "- УК (Управляющая Компания) → Verwaltungsgesellschaft\n"
            "- Международная Компания (International Company) → Internationale Gesellschaft\n"
            "- МКАО (Международная Компания Акционерное Общество) → Internationale Aktiengesellschaft "
            "(NEVER 'Internationale Gesellschaft der Aktiengesellschaft' — that is redundant and awkward)\n"
            "- Общество (as used in Russian documents for 'the Company') → Gesellschaft (use CONSISTENTLY)\n"

            "COMMITTEES AND BODIES:\n"
            "- Технический комитет / Technical Committee → Technischer Ausschuss (TA) — "
            "Pick ONE abbreviation ('TA') and use it CONSISTENTLY throughout. "
            "NEVER switch between 'TA' and 'TK' within the same document.\n"
            "- Наблюдательный совет / Supervisory Board → Aufsichtsrat\n"

            "REAL ESTATE AND LEASE TERMS:\n"
            "- lease → Mietvertrag / Pachtvertrag\n"
            "- rent → Miete\n"
            "- premises → Mietobjekt / Räumlichkeiten\n"
            "- nuisance → Belästigung / Störung\n"
            "- shareholding / equity stake → Beteiligung / Kapitalanteil\n"
            "- remedies → Rechtsbehelfe / Rechtsmittel (NEVER 'Heilmittel')\n"
            "- written notice → schriftliche Mitteilung / schriftliche Kündigung\n"
            "- 'three (3) months notice' → 'mit einer Frist von drei (3) Monaten'\n"
            "- 'it is hereby agreed as follows' → 'ES WIRD HIERMIT WIE FOLGT VEREINBART'\n"
            "- act of God → höhere Gewalt\n"
            "- security deposit → Kaution / Sicherheitsleistung\n"

            "ARCHITECTURAL AND BUILDING TERMS:\n"
            "- basement / underground floor → Untergeschoss / Kellergeschoss\n"
            "- mezzanine / mezzanine floor → Zwischengeschoss / Mezzanin\n"
            "- ground floor → Erdgeschoss\n"
            "- floor plan → Grundriss\n"

            "FINANCIAL AND LEGAL TERMS:\n"
            "- расчеты / settlements → Zahlungen / Abrechnungen (NEVER 'Berechnungen' or 'Kalkulationen')\n"
            "- financial assistance → finanzielle Unterstützung (NEVER 'finanzielle Hilfe' in legal context)\n"
            "- Договор займа → Darlehensvertrag (note the genitive 's': DarlehenSvertrag)\n"
            "- Договор цессии → Abtretungsvertrag / Zessionsvertrag\n"
            "- Устав → Satzung / Gesellschaftsvertrag\n"
            "- Доверенность → Vollmacht\n"
            "- Протокол → Protokoll / Niederschrift\n"
            "- Решение → Beschluss / Entscheidung\n"
            "- по решению / по усмотрению → nach Ermessen von / durch Beschluss von\n"
            "- прошито и пронумеровано → geheftet und nummeriert\n"
            "- прошито, пронумеровано, опечатано → geheftet, nummeriert und mit Dienstsiegel versehen "
            "(NEVER 'versiegelt' — 'опечатано' means stamped with official seal, not 'sealed')\n"
            "- gutgeschrieben (credited) → use 'angerechnet' when the meaning is 'applied towards an obligation'\n"
            "- professional trustee → professioneller Treuhänder\n"

            "ABBREVIATION CONSISTENCY RULES (CRITICAL):\n"
            "- When introducing an abbreviation, ALWAYS define it in parentheses on first use.\n"
            "- Use the SAME abbreviation for the SAME concept throughout the entire document.\n"
            "- геологическое изучение → geologische Untersuchung (GU) — use 'GU' consistently\n"
            "- геологоразведочные работы → geologische Erkundungsarbeiten (GEA) — use 'GEA' consistently\n"
            "- Технический комитет → Technischer Ausschuss (TA) — use 'TA' consistently "
            "(NEVER switch between 'TA' and 'TK')\n"
            "- NEVER use different abbreviations for the same concept within one document.\n"

            "RUSSIAN INSTITUTIONS:\n"
            "- ОВД (Отдел Внутренних Дел) → Polizeidienststelle / Abteilung für Innere Angelegenheiten\n"
            "- ЗАГС → Standesamt\n"
            "- ИНН → Steueridentifikationsnummer (Steuer-IdNr.)\n"
            "- ОГРН → Staatliche Registrierungsnummer\n"
            "- КПП → Registrierungsgrundcode\n"
            "- БИК → Bankidentifikationscode\n"
            "- Расчетный счет / Р/С → Girokonto (Konto-Nr.)\n"
            "- Корреспондентский счет / К/С → Korrespondenzkonto\n"

            "RUSSIAN NAME TRANSLITERATION FOR GERMAN:\n"
            "- Patronymics: use the NOMINATIVE form consistently. "
            "Example: 'Суровая Елена Борисовна' → 'Surowaja Elena Borissowna' (ALWAYS the same form, "
            "NEVER switch between 'Surowaja' and 'Surowa' within the same document).\n"
            "- When a Russian name appears in instrumental/genitive case (e.g. 'Суровой Еленой Борисовной'), "
            "transliterate the NOMINATIVE form, not the declined form.\n"

            "GENERAL RULES FOR GERMAN:\n"
            "- Use standard High German (Hochdeutsch) — NEVER Austrian German, Swiss German, or dialect\n"
            "- Use formal legal register: use 'hat zu' or 'ist verpflichtet' for obligations, 'hiermit' for declarations\n"
            "- Use standard German date format: '15. Oktober 2025'\n"
            "- Maintain formal legal phrasing: 'vertreten durch den Geschäftsführer', 'handelnd aufgrund von'\n"
            "- ALL English words MUST be translated — do NOT leave any English terms in the output "
            "(except proper nouns, company names, and internationally recognized abbreviations)\n"
            "- Use proper German characters: ä, ö, ü, ß (Eszett). NEVER replace ß with 'ss' in legal documents\n"
            "- Capitalize ALL nouns as per German grammar rules — NEVER write nouns in lowercase\n"
            "- GENITIVE CASE: Pay strict attention to genitive endings. "
            "Examples: 'des Staatsbürgers' (NOT 'des Staatsbürger'), 'des Quorums' (NOT 'der Quorum'), "
            "'eines Darlehensvertrags' (NOT 'eines Darlehensvertrag')\n"
            "- GENDER: Pay strict attention to grammatical gender. "
            "Examples: 'das Quorum' (neuter), 'die Satzung' (feminine), 'der Beschluss' (masculine)\n"
            "- LONG SENTENCES: Break overly long sentences into shorter, well-punctuated German clauses. "
            "Do NOT copy English sentence structure when it produces incomprehensible German. "
            "Ensure every sentence has a complete verb phrase — NEVER leave infinitive constructions hanging.\n"
            "- VERB PLACEMENT: In German subordinate clauses, the conjugated verb goes to the END. "
            "In main clauses, the conjugated verb is in SECOND position. Do NOT copy English word order.\n"
            "- REFLEXIVE VERBS: Always include the reflexive pronoun. "
            "Example: 'sich versichern' — NEVER omit 'sich'.\n"
            "- When the source text is poorly drafted (awkward, ambiguous, or ungrammatical), "
            "translate the INTENDED MEANING into natural, correct German. Do NOT produce a word-for-word calque "
            "that reproduces the source's drafting problems.\n"
        )

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
# Load model with vLLM — auto-detects GPU
# =====================================================
def load_model():
    global llm_engine, tokenizer
    if llm_engine is not None:
        return

    # IMPORTANT: Do NOT call torch.cuda.* before vLLM init!
    # It initializes CUDA which forces 'spawn' multiprocessing and crashes.
    log("Loading model with vLLM engine...")
    t0 = time.time()

    llm_engine = LLM(
        model=MODEL_PATH,
        dtype="auto",                    # auto-selects BF16 on Ampere+
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        trust_remote_code=True,
        enable_prefix_caching=True,       # Caches system prompt KV across pages
    )

    tokenizer = llm_engine.get_tokenizer()

    # Log GPU info AFTER vLLM has initialized CUDA
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    log(f"vLLM engine ready in {time.time()-t0:.1f}s")


# =====================================================
# Helper: build prompt from messages
# =====================================================
def build_prompt(messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n/nothink\n")
        return "\n".join(parts)


# =====================================================
# Layout / OCR helpers
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

def clean_ocr_noise(text: str) -> str:
    cleaned, seen = [], set()
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or is_layout_line(line):
            continue
        if len(re.findall(r"[^\W\d_]", line, re.UNICODE)) < 5:
            continue
        upper = line.upper()
        if upper in seen:
            continue
        seen.add(upper)
        cleaned.append(line)
    return "\n".join(cleaned)

def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    if truncated.rstrip().endswith("."):
        return truncated.rstrip()
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    last_excl = truncated.rfind("! ")
    last_quest = truncated.rfind("? ")
    best = max(last_period, last_excl, last_quest)
    if best > len(truncated) * 0.6:
        return truncated[:best + 1].strip()
    return truncated.rstrip()

def clean_output(decoded: str) -> str:
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()
    for marker in ["STRICT RULES:", "LEGAL TRANSLATION STANDARDS:"]:
        if marker in decoded:
            idx_m = decoded.find(marker)
            lines = decoded[idx_m:].split("\n")
            last_rule = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("- "):
                    last_rule = i
            decoded = "\n".join(lines[last_rule + 1:]).strip()
    return decoded


# =====================================================
# Token limits and helpers for context-aware chunking
# =====================================================
MAX_PROMPT_TOKENS = 12000  # Leave headroom below 16384 for generation + template overhead

def _estimate_tokens(text):
    """Rough token estimate: ~1 token per 3.5 characters for mixed content."""
    return len(text) // 3


# =====================================================
# TRANSLATION — vLLM parallel batch
# =====================================================
def translate_text_batch(texts, target_language="English"):
    translate_prompt = build_translate_prompt(target_language)

    prompts = []
    valid_indices = []
    results = [""] * len(texts)

    for idx, text in enumerate(texts):
        stripped = (text or "").strip()
        if not stripped or len(re.findall(r"[^\W\d_]", stripped, re.UNICODE)) < 5:
            results[idx] = text or ""
            continue

        messages = [
            {"role": "system", "content": translate_prompt},
            {"role": "user", "content": stripped}
        ]
        prompt = build_prompt(messages)

        # Safety check: if a single page exceeds context, split it
        if _estimate_tokens(prompt) > MAX_PROMPT_TOKENS:
            words = stripped.split()
            mid = len(words) // 2
            for half in [" ".join(words[:mid]), " ".join(words[mid:])]:
                half_msgs = [
                    {"role": "system", "content": translate_prompt},
                    {"role": "user", "content": half}
                ]
                prompts.append(build_prompt(half_msgs))
                valid_indices.append(idx)  # both halves map to same index
        else:
            prompts.append(prompt)
            valid_indices.append(idx)

    if not prompts:
        return results

    log(f"Translating {len(prompts)} pages in parallel with vLLM...")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
    )

    t0 = time.time()
    outputs = llm_engine.generate(prompts, sampling_params)
    gen_time = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    log(f"Translation: {total_tokens} tokens in {gen_time:.1f}s "
        f"({total_tokens/gen_time:.1f} tok/s effective)")

    for i, output in enumerate(outputs):
        translated = clean_output(output.outputs[0].text)
        idx = valid_indices[i]
        if results[idx]:
            results[idx] += "\n" + translated  # Concatenate split-page halves
        else:
            results[idx] = translated

    return results


# =====================================================
# SUMMARY — chunked to fit context window
# =====================================================

def _build_summary_prompt(text_block, target_words, system_prompt):
    """Build a summary prompt from a text block and return the formatted string."""
    user_content = (
        f"Summarize the following document in approximately {target_words} words. "
        f"Make sure to complete all sentences properly.\n\n"
        f"DOCUMENT:\n{text_block}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return build_prompt(messages)

def _chunk_pages_by_tokens(cleaned_pages, max_tokens):
    """Split cleaned page texts into chunks that fit within max_tokens."""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for page_text in cleaned_pages:
        page_tokens = _estimate_tokens(page_text)
        # If a single page exceeds the limit, truncate it
        if page_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            # Truncate to fit
            char_limit = max_tokens * 3
            chunks.append(page_text[:char_limit])
            continue

        if current_tokens + page_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(page_text)
        current_tokens += page_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def summarize_all_pages(pages, max_words, system_prompt):
    # Collect and clean all page texts
    cleaned_pages = []
    for p in pages:
        cleaned = clean_ocr_noise(p["text"])
        if cleaned and len(re.findall(r"[^\W\d_]", cleaned, re.UNICODE)) > 20:
            cleaned_pages.append(cleaned)

    if not cleaned_pages:
        log("ERROR: No valid text found for summary")
        return ""

    full_text = "\n\n".join(cleaned_pages)
    doc_word_count = len(full_text.split())
    actual_target = max(50, min(max_words, doc_word_count // 3))
    log(f"Summary target: {actual_target} words (doc has {doc_word_count} words)")

    # Check if the full text fits in one prompt
    test_prompt = _build_summary_prompt(full_text, actual_target, system_prompt)
    prompt_tokens = _estimate_tokens(test_prompt)

    if prompt_tokens <= MAX_PROMPT_TOKENS:
        # Single-shot: fits in context
        log("Summary: single-shot (fits in context)")
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=min(actual_target * 5, 4096),
        )
        t0 = time.time()
        outputs = llm_engine.generate([test_prompt], sampling_params)
        gen_time = time.time() - t0
        decoded = clean_output(outputs[0].outputs[0].text)
        result = limit_words(decoded, actual_target)
        log(f"Summary: {len(result.split())} words in {gen_time:.1f}s")
        return result

    # Chunked summarization: split pages into token-safe groups
    chunks = _chunk_pages_by_tokens(cleaned_pages, MAX_PROMPT_TOKENS)
    log(f"Summary: document too large, splitting into {len(chunks)} chunks")

    # Phase 1: Summarize each chunk
    words_per_chunk = max(100, actual_target // len(chunks) + 50)
    chunk_prompts = []
    for i, chunk_text in enumerate(chunks):
        chunk_prompts.append(
            _build_summary_prompt(chunk_text, words_per_chunk, system_prompt)
        )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=min(words_per_chunk * 5, 4096),
    )

    t0 = time.time()
    chunk_outputs = llm_engine.generate(chunk_prompts, sampling_params)
    phase1_time = time.time() - t0
    log(f"Summary phase 1: {len(chunks)} chunks summarized in {phase1_time:.1f}s")

    chunk_summaries = []
    for output in chunk_outputs:
        chunk_summaries.append(clean_output(output.outputs[0].text))

    # Phase 2: Combine chunk summaries into final summary
    combined = "\n\n".join(
        f"[Part {i+1}]: {s}" for i, s in enumerate(chunk_summaries)
    )

    combine_user = (
        f"Below are summaries of different sections of a single document. "
        f"Combine them into ONE coherent summary of approximately {actual_target} words. "
        f"Make sure to complete all sentences properly. "
        f"Do NOT list the parts separately — write a single unified paragraph.\n\n"
        f"{combined}"
    )
    combine_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combine_user}
    ]
    combine_prompt = build_prompt(combine_messages)

    sampling_params_final = SamplingParams(
        temperature=0,
        max_tokens=min(actual_target * 5, 4096),
    )

    t0 = time.time()
    final_outputs = llm_engine.generate([combine_prompt], sampling_params_final)
    phase2_time = time.time() - t0

    decoded = clean_output(final_outputs[0].outputs[0].text)
    result = limit_words(decoded, actual_target)

    log(f"Summary: {len(result.split())} words in {phase1_time + phase2_time:.1f}s total "
        f"(phase1={phase1_time:.1f}s, phase2={phase2_time:.1f}s)")
    return result


# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    input_data = event["input"]
    pages = input_data["pages"]
    max_words = int(input_data.get("n_words", 500))
    system_prompt = input_data.get("system_prompt", DEFAULT_SUMMARY_PROMPT)
    target_language = input_data.get("target_language", "English")

    log(f"Processing {len(pages)} pages, target: {max_words} words, translate to: {target_language}")

    load_model()

    # 1) Translate all pages in parallel
    log(f"Starting batch translation to {target_language}...")
    start = time.time()
    page_texts = [p["text"] for p in pages]
    translated_texts = translate_text_batch(page_texts, target_language)
    for i, p in enumerate(pages):
        p["text"] = translated_texts[i]
    log(f"Translation done in {time.time()-start:.2f}s")

    # 2) Summarize
    log(f"Creating summary ({max_words} words)")
    start = time.time()
    summary = summarize_all_pages(pages, max_words, system_prompt)
    log(f"Summary done in {time.time()-start:.2f}s")

    if not summary:
        log("WARNING: Summary is empty!")

    log("Handler finished")
    return {"summary": summary, "pages": pages}

runpod.serverless.start({"handler": handler})
