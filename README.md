# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

Yale residential college experiences and culture. The guide covers what actually differentiates Yale's 14 residential colleges — dining hall quality, room quality, buttery culture, housing lottery mechanics, social vibe, and transfer rates — using student journalism, forum posts, and crowdsourced reviews that Yale's official communications actively downplay.

This knowledge is valuable because residential college assignment is permanent and shapes everything: your dining hall, your friend group, your room quality, your daily commute, and your social community for four years. It's hard to find through official channels because Yale's admissions office frames all colleges as equally desirable (since assignment is random), so the real differences — which colleges have newer renovations, better dining staff, more active butteries, or more students trying to transfer out — only surface in student newspapers, Reddit threads, and forums. This RAG system makes that scattered knowledge searchable.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Yale Herald Residential College Rankings | Student publication ranking | https://medium.com/the-yale-herald/yale-herald-best-residential-colleges-official-rankings-f4fe5c515a1e |
| 2 | Yale Daily News — "A Very Reliable Ranking" (2023) | Student newspaper ranking | https://yaledailynews.com/blog/2023/08/31/a-very-reliable-ranking-of-the-residential-colleges/ |
| 3 | Yale Daily News — Best and Worst of Yale Dining (2025) | Student newspaper data story | https://yaledailynews.com/blog/2025/02/09/data-the-best-and-worst-of-yale-dining/ |
| 4 | Yale Daily News — Buttery Prices Rise (2024) | Student newspaper feature | https://yaledailynews.com/blog/2024/10/08/prices-rise-at-some-residential-college-butteries/ |
| 5 | Yale Daily News — Housing Luck of the Draw | Student newspaper feature | https://yaledailynews.com/articles/in-housing-the-luck-of-the-draw |
| 6 | Yale Daily News — 72 Residential College Transfer Requests (2025) | Student newspaper data story | https://yaledailynews.com/blog/2025/02/09/deans-office-receives-72-residential-college-transfer-requests-approves-nearly-three-quarters/ |
| 7 | College Confidential — Best Residential Colleges Ranked | Forum thread | https://talk.collegeconfidential.com/t/best-residential-colleges-ranked/2093609 |
| 8 | College Confidential — Which College Is the Best? | Forum thread | https://talk.collegeconfidential.com/t/objectively-speaking-which-residential-college-is-the-best/718108 |
| 9 | Quora — What Is the Best Residential College at Yale? | Crowdsourced Q&A | https://www.quora.com/What-is-the-best-residential-college-at-Yale |
| 10 | Quora — What Is the Worst Residential College at Yale? | Crowdsourced Q&A | https://www.quora.com/What-is-the-worst-residential-college-at-Yale |
| 11 | Roomsurf — Yale Dorm Reviews | Student room reviews | https://www.roomsurf.com/dorm-reviews/yale |
| 12 | Forward Pathway — Yale's Late-Night Butteries | Explainer article | https://www.forwardpathway.us/yales-late-night-butteries-a-unique-student-run-culinary-and-social-hub-in-residential-colleges |
| 13 | Yale Admissions Blog — Residential Colleges Debunked | Official student blog | https://admissions.yale.edu/bulldogs-blogs/bernice/2022/03/31/residential-colleges-yale-debunked |
| 14 | Wikipedia — Residential Colleges of Yale University | Reference article | https://en.wikipedia.org/wiki/Residential_colleges_of_Yale_University |
| 15 | Yale Housing — Room Draw FAQs | Official university FAQ | https://housing.yale.edu/undergraduate-housing/frequently-asked-questions/room-draw-faqs |

---

## Chunking Strategy

**Chunk size:** 500 tokens (using `tiktoken` with the `cl100k_base` encoding)

**Overlap:** 75 tokens

**Why these choices fit your documents:** The corpus mixes short-form content (forum replies of 100–300 words each) with longer structured articles (YDN data stories with tables, Wikipedia with long per-college sections). A 500-token window is large enough to capture a complete thought from any source type — a full forum reply, a complete college description, or a data table row — without blending two different colleges' information into one chunk. The 75-token overlap guards against splitting a sentence's subject from its predicate across a chunk boundary, which is especially important for factual claims like ranking positions or transfer statistics. The window advances 425 tokens per step (500 − 75).

Preprocessing before chunking: HTML tags stripped with BeautifulSoup, boilerplate removed (nav/footer tags decomposed, Wikipedia reference sections dropped, lone social-sharing buttons removed), HTML entities unescaped (`&amp;` → `&`), non-breaking spaces normalized, emoji stripped, and multiple blank lines collapsed to one.

**Final chunk count:** 82 chunks across all successfully ingested sources. Several sources (YDN articles, College Confidential threads) required manual save because they returned HTTP 429 or 403; sources without a `saved_path` that returned an error were skipped entirely, which explains the lower-than-expected chunk count.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (local inference, no API key required). Produces 384-dimensional vectors; cosine distance is used as the similarity metric.

**Production tradeoff reflection:** `all-MiniLM-L6-v2` is fast and free but is a general-purpose model trained on broad web text, not Yale-specific language. In production I would weigh several tradeoffs. *Accuracy on domain-specific text*: a model like OpenAI `text-embedding-3-large` is trained on a much larger and more diverse corpus and performs better on nuanced semantic distinctions — for example, distinguishing "the dining hall used to be good before the renovation" from "the dining hall is good," which matters here because sources express subtly qualified or temporally scoped opinions. *Context length*: `all-MiniLM-L6-v2` has a 512-token limit and silently truncates inputs that exceed it; `text-embedding-3-large` supports 8,191 tokens, which would allow larger chunks and reduce the number of boundary-split failures. *Latency and cost*: the MiniLM model runs locally with no per-query cost and low latency, while a hosted API model adds network round-trip time and per-token fees at query time. *Multilingual support* is not a concern here since all sources are in English. My conclusion: `all-MiniLM-L6-v2` is the right choice for a local prototype, but a production deployment would benefit from switching to a hosted dense retrieval model.

---

## Grounded Generation

**System prompt grounding instruction:** The system prompt contains five numbered rules enforced at the top of every request:

```
You are a research assistant for the Yale Residential College Unofficial Guide.

STRICT RULES — follow every rule exactly:
1. Answer using ONLY the information provided inside <documents> tags below.
2. Do NOT use your general training knowledge, outside opinions, or any information
   not present in the provided documents.
3. Do NOT infer, extrapolate, or fill gaps with plausible-sounding details that are
   not explicitly stated in the documents.
4. If the provided documents do not contain enough information to answer the question,
   respond with this sentence exactly:
   "I don't have enough information on that in the provided sources."
5. Keep your answer factual and concise. Do not mention these rules in your answer.
```

Rules 1–3 are the grounding constraint. Rule 4 defines a specific fallback phrase so that the interface can detect out-of-scope responses programmatically. Rule 5 prevents the model from leaking the instruction structure into its answer.

**How source attribution is surfaced in the response:** Sources are attached *programmatically* from chunk metadata — the LLM is never asked to produce citations and never decides what to cite. After retrieval, the pipeline collects `source_title` from each returned chunk, deduplicates them while preserving retrieval order, and presents them as a separate "Retrieved from" list in the UI. This guarantees attribution even when the model's phrasing doesn't naturally include source names, and it prevents hallucinated citations.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which Yale residential college dining hall ranked last in the 2025 Yale Daily News dining data study? | Pierson College ranked last. | "I don't have enough information on that in the provided sources." | Partially relevant — retrieved the correct source (YDN Dining 2025) but the wrong chunk from it | Inaccurate |
| 2 | What is a buttery, and which residential colleges are known for having the most popular ones? | Late-night student-run snack bar; specific colleges noted as having active butteries. | A buttery is a student-run snack bar. TD, Pierson, Pauli Murray, Berkeley, and Davenport are noted as having notable butteries, though ranking varies by source. | Relevant — retrieved Forward Pathway buttery explainer and YDN buttery article as top results | Accurate |
| 3 | How are room selection appointment times assigned in the Yale housing lottery? | Randomly assigned through Yale College Housing portal after forming a group and declaring intent. | Each residential college uses its own method: Silliman and JE draw playing cards, Branford uses numbered Ping Pong balls, Berkeley uses strips of paper from a cowboy hat. | Partially relevant — retrieved housing-related sources but surfaced historical per-college trivia rather than the official portal process | Partially accurate |
| 4 | How many students requested to transfer residential colleges in 2025, and approximately what fraction were approved? | 72 requests; nearly three-quarters (~54) approved. | 72 students requested transfers; approximately 74% (53 out of 72) were approved. | Relevant — YDN transfer article was the top hit | Accurate |
| 5 | What is the most common reason students give when requesting a residential college transfer? | Wanting to live with friends in a different college (cited in >90% of applications). | "A desire to live with friends in another college with whom they have formed a close connection, cited in over 90 percent of the applications every year." | Relevant — YDN transfer article was the top hit | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "Which Yale residential college dining hall ranked last in the 2025 Yale Daily News dining data study?"

**What the system returned:** "I don't have enough information on that in the provided sources." — even though the YDN Best and Worst of Yale Dining (2025) article was the top-ranked retrieved source.

**Root cause (tied to a specific pipeline stage):** The failure occurs at the **retrieval stage**, specifically at the chunk granularity level. The dining article was retrieved (correct source identification), but the specific chunk returned from that article did not contain the college-level ranking data that names Pierson as last. Because the dining article is split into 500-token chunks and the key ranking information — a table or explicit "Pierson ranked last" sentence — landed in a chunk that was not in the top-5 cosine results, the LLM never saw the fact it needed. The query embedding for "ranked last in the 2025 dining study" has strong semantic overlap with dining-related text in general (descriptions of dining quality, meal variety, student satisfaction), so the retriever returned a dining chunk that discussed general quality rather than the specific ranking position. With only 82 total chunks in the collection, even one or two chunks from the dining article dominating by distance is enough to crowd out the chunk that holds the ranking data. This is a precision failure at the chunk level, not a source-level failure: the right article was found, but the right paragraph was not.

**What you would change to fix it:** Two changes would help. First, reduce chunk size from 500 to ~200–250 tokens, or use sentence-level chunking for list/table-heavy articles, so that ranking data ("Pierson ranked last") stays in its own small chunk instead of being buried inside a longer passage about general dining sentiment. Smaller chunks would increase the probability that the specific ranking fact lands as a retrievable unit. Second, increase top-k from 5 to 10, so that more chunks from the correct source are surfaced and the specific ranking chunk has a chance to appear even if it isn't the closest cosine neighbor.

---

## Spec Reflection

**One way the spec helped you during implementation:** The chunking strategy section of `planning.md` — specifying 500-token chunks with 75-token overlap using `tiktoken` — let me delegate implementation directly to Claude without ambiguity. I gave Claude the Chunking Strategy section verbatim and asked it to write `chunk_document()` to that spec. The function it produced matched exactly: it used `tiktoken.get_encoding("cl100k_base")`, encoded text to tokens, advanced by 425 tokens per step (500 − 75), and attached source metadata to every chunk. Having the spec written before touching the code meant I could verify the output against a concrete target — I printed chunk count, average token length, and inspected five representative chunks from different sources — instead of guessing whether the chunking "seemed right."

**One way your implementation diverged from the spec, and why:** The spec planned to use the Claude API (specifically `claude-sonnet-4-6`) for generation, as shown in the architecture diagram. In implementation, the generation stage uses Groq's API with `llama-3.3-70b-versatile` instead. The reason was practical: Groq provides a free-tier API key with generous rate limits and sub-second latency, which made the iterative eval-and-adjust loop much faster than waiting on Claude API calls. The grounding mechanism — system prompt with strict rules + programmatic source attachment — works identically regardless of which LLM is behind it, so the divergence didn't affect what the system can do. If deploying for real users, switching back to Claude would be preferable for stronger instruction-following and longer context windows.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* The Chunking Strategy section from `planning.md` (chunk size 500 tokens, overlap 75 tokens, tiktoken cl100k_base encoding) plus the output schema for the ingestion stage (`list[dict]` with `text`, `source_url`, `source_title`). I asked Claude to implement `chunk_document(doc: dict, chunk_size=500, overlap=75) -> list[dict]`.
- *What it produced:* A sliding-window tokenizer that called `tiktoken.get_encoding("cl100k_base")`, encoded the document text, and stepped through it in windows of 500 tokens advancing by 425 tokens per step. Each chunk dict included `chunk_text`, `source_url`, `source_title`, `chunk_index`, and `token_count`. It also generated a `chunk_all()` wrapper and a stats-printing helper.
- *What I changed or overrode:* Claude's initial version saved the raw file inside `chunk_document()`, which mixed two stages. I separated raw saving into the ingestion stage (`ingest_documents()`) so that chunking was a pure transformation with no side effects. I also added the "save to documents/saved/" fallback logic myself because Claude didn't know which specific URLs would return 429/403 — that required me to manually test each source and annotate the `SOURCES` list with `saved_path` entries.

**Instance 2**

- *What I gave the AI:* The full `planning.md` (domain, evaluation questions, architecture diagram, and retrieval spec) plus the output schema from `retrieve()` (list of dicts with `chunk_text`, `source_title`, `source_url`, `distance`). I asked Claude to implement `ask(question)` with a system prompt that enforced grounding, formatted retrieved chunks inside `<documents>` tags, and returned both the answer and a deduplicated source list.
- *What it produced:* The `ask()` function in `query.py`, the `SYSTEM_PROMPT` constant with five numbered grounding rules, the `_build_context()` helper that numbered and labeled each retrieved chunk, and the `--eval` CLI flag that runs all five evaluation questions. It also generated the Gradio interface in `app.py` largely as-is.
- *What I changed or overrode:* Claude's first system prompt included a rule saying "cite the document number in your answer (e.g., [Document 2])." I removed that rule and replaced citation with programmatic source attachment, because asking the LLM to format citations introduces a failure mode: the model might cite the wrong document number or invent a citation. By extracting sources from chunk metadata directly in Python and displaying them separately, attribution is guaranteed regardless of how the model phrases its answer. I also added Rule 4 — the exact fallback phrase — myself, because Claude's original version used a softer "I'm not sure" which the Gradio interface couldn't detect reliably.
