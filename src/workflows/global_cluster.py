from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from src.models.provider_factory import LLMProviderFactory

from utils.utils import chunk_awards, parse_json_from_llm, save_employee_signals, save_clustering_result

class GlobalCluster:

    def __init__(self, provider, model, temperature, api_key):
        self.llm = LLMProviderFactory.create(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=4000
        )

    
    def extract_raw_signals(self, employee) -> str:
        rec_id = employee.get("rec_id")
        awards = employee.get("awards")

        award_chunks = chunk_awards(rec_id=rec_id, awards_list=awards)

        def process_chunk(chunk_text):
            prompt = self._build_extracting_signal_prompt(chunk_text)
            raw = self.llm.call(prompt)
            print(" response:", raw)

            try:
                return parse_json_from_llm(raw)
            except Exception as e:
                print("JSON PARSE ERROR:", e)
                return {} 

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(process_chunk, award_chunks[0:3])

        all_results = {}
        signal_set = set()

        for id, r in enumerate(results):
            print(f'############### {id} #################')
            # skip invalid or empty
            if not r or not isinstance(r, dict):
                continue

            for award_index, sentence_dict in r.items():
                award_index = str(award_index)

                if not isinstance(sentence_dict, dict):
                    print(f"[WARN] Invalid sentence_dict for award {award_index}: {sentence_dict}")
                    continue

                all_results[award_index] = sentence_dict

        save_employee_signals(rec_id=rec_id, results=all_results)

        for award_index, sentence_dict in all_results.items():
            for sentence_index, signals in sentence_dict.items():
                for s in signals:
                    signal_set.add(s)


        return rec_id, all_results




    def clustering_signal(self, rec_id, signal_set):
        prompt = self._build_cluster_prompt(signal_set)
        raw = self.llm.call(prompt)

        try:
            parsed_json = parse_json_from_llm(raw)
        except Exception as e:
            print("JSON PARSE ERROR:", e)
            prased_json = {}

        save_clustering_result(rec_id, parsed_json)

        return parsed_json

    def _build_extracting_signal_prompt(self, chunk_text):
        return f"""
                You will analyze multiple award entries from a single employee.

                Each award entry follows the format:
                <award_index># <title> | <message>

                Your goal is to extract ONLY meaningful, promotion-relevant behavioral signals
                from semantic chunks. Do NOT force extraction when meaning is weak or absent.

                --------------------------------------------------------
                STEP 1 — SEMANTIC CHUNKING
                --------------------------------------------------------
                Do NOT split by sentence. Instead:

                1. Read the <title> and <message> entirely.
                2. Break text into **semantic chunks**:
                - Each chunk must contain ONE complete behavioral idea.
                - A chunk may span part of a sentence, a full sentence, or multiple sentences.
                3. Chunk order is preserved.

                Chunking is internal only and should NOT appear in the final JSON.

                --------------------------------------------------------
                STEP 2 — PHRASE-LEVEL SIGNAL CANDIDATES
                --------------------------------------------------------
                For each semantic chunk:
                - Attempt to extract **0–2 behavioral signals** as short phrases (1–4 words).
                - A phrase should represent a **distinct, meaningful, promotable competency**.

                IMPORTANT:
                - If a chunk is weak, generic, descriptive, or has no promotable meaning:
                    → Extract **0** signals.
                - Do NOT force extraction.
                - Do NOT convert generic praise ("thanks", "great job", "hard work") into signals.

                Valid signals typically reflect:
                leadership, strategic impact, influence, cross-functional work, innovation,
                mentorship, initiative ownership, stakeholder alignment, complex problem-solving,
                organizational impact, execution excellence, vision-setting.

                Weak signals to exclude:
                basic teamwork, generic positivity, routine duties, vague compliments.

                --------------------------------------------------------
                STEP 3 — VP PROMOTION RELEVANCE FILTER
                --------------------------------------------------------
                Keep ONLY signals that meaningfully support a **VP-level promotion case**.

                Discard signals unless they match one of the following:
                - strategic leadership
                - organizational impact
                - cross-team or cross-functional influence
                - mentorship / people development
                - high-stakes execution
                - complex problem solving
                - initiative / ownership
                - innovation leadership
                - decision-making influence
                - stakeholder or executive alignment
                - change leadership
                - long-term vision or planning

                If a candidate signal is **not clearly helpful to a VP promotion**, discard it.

                --------------------------------------------------------
                STEP 4 — INDEXING RULES
                --------------------------------------------------------
                Use original award_index.

                Within each award:
                - chunk_index "0" = title chunk (only if title is meaningful).
                - chunk_index "1", "2", ... = semantic chunks from the message.

                If a chunk produces 0 signals → omit that chunk entirely.

                --------------------------------------------------------
                STEP 5 — OUTPUT SPECIFICATION
                --------------------------------------------------------
                Return ONE JSON object:

                {{
                "<award_index>": {{
                    "<chunk_index>": ["signal1", "signal2"],
                    ...
                }},
                ...
                }}

                - Keys must be strings.
                - Values are arrays of phrase-level signals.
                - Do NOT output any explanation, text, commentary, or chunk content.

                --------------------------------------------------------
                FEW-SHOT EXAMPLES (STRICTLY FOLLOW THIS BEHAVIOR)
                --------------------------------------------------------

                Example Input:
                0# Innovation Spotlight | She created a new automation. It saved 40 hours. The change was well received.

                Example Output:
                {{
                "0": {{
                    "0": ["innovation"],
                    "1": ["process automation"],
                    "2": ["efficiency improvement"]
                }}
                }}

                Explanation: “well received” is discarded because it is not VP-relevant.

                --------------------------------------------------------

                Example Input:
                0# Recognition Award | Thank you for your hard work. You always give your best.

                Example Output:
                {{
                "0": {{
                    "2": ["dedication"]
                }}
                }}

                Explanation:
                - Title not meaningful → skip.
                - Sentence 1 → no promotable meaning → skip.
                - Sentence 2 → “dedication” is meaningful → keep.

                --------------------------------------------------------

                Example Input:
                0# Team Leadership Award | She led a cross-functional migration effort. She coordinated directors and ICs across multiple regions.

                Example Output:
                {{
                "0": {{
                    "0": ["leadership"],
                    "1": ["cross-functional leadership"],
                    "2": ["multi-level coordination"]
                }}
                }}

                --------------------------------------------------------

                NOW PROCESS THE FOLLOWING AWARDS:
                {chunk_text}

                Return ONLY the JSON object.
            """

    
    def _build_cluster_prompt(self, phrase_list):
        

        return f"""
            You will perform semantic clustering on a list of behavior phrases.
            Your goal is to group similar phrases into meaningful VP-level behavioral themes.

            ============================================================
            STEP 1 — UNDERSTAND THE INPUT
            ============================================================
            You are given a list of short behavior phrases extracted from award texts.
            Each phrase expresses a promotable competency such as:
            leadership, strategic thinking, organizational impact, influence,
            mentorship, client ownership, execution excellence, etc.

            Your job is not to evaluate individuals, but to discover
            the underlying BEHAVIOR PATTERNS shared across the list.

            The number of clusters is NOT fixed.
            Only create clusters when a meaningful theme exists.

            ============================================================
            STEP 2 — REMOVE NOISE OR LOW-VALUE PHRASES
            ============================================================
            Before clustering:
            - Discard duplicate phrases.
            - Discard near-duplicates that convey identical meaning.
            - Discard weak phrases that do NOT represent promotable behaviors,
            such as:
                • basic cooperation
                • simple task execution
                • vague positivity
                • generic praise (“great work”, “good effort”)

            Only keep phrases that are:
            - promotable,
            - competency-based,
            - and VP-relevant.

            ============================================================
            STEP 3 — SEMANTIC CLUSTERING
            ============================================================
            Cluster phrases based on their underlying behavioral meaning.

            Rules:
            1. Phrases inside a cluster must share a coherent behavioral concept.
            2. Clusters must be distinct from one another.
            3. Do NOT force a phrase into a cluster if it doesn’t belong.
            4. If a phrase does not fit any group, create a single-phrase cluster.

            Good cluster themes include:
            - cross-functional leadership
            - strategic communication
            - stakeholder influence
            - organizational impact
            - client ownership
            - innovation leadership
            - people development
            - high-stakes execution
            - vision & planning
            - crisis leadership
            - complex problem-solving

            ============================================================
            STEP 4 — NAME EACH CLUSTER
            ============================================================
            For each cluster:
            - Create a short (1–4 word) descriptive cluster name.
            - The name should reflect the core behavior theme.
            - Names must be at the VP/leadership competency level.

            Examples:
            - “Cross-functional Leadership”
            - “Stakeholder Influence”
            - “Execution Ownership”
            - “Strategic Communication”
            - “Client Partnership Leadership”
            - “People Development”

            Do NOT generate vague or generic labels.

            ============================================================
            STEP 5 — EXPLAIN EACH CLUSTER
            ============================================================
            For each cluster, write a **one-sentence explanation** describing:
            - what this behavior represents,
            - and why it matters for senior leadership / VP roles.

            Example:
            “This theme reflects the ability to align diverse stakeholders across organizations to drive complex initiatives.”

            ============================================================
            STEP 6 — OUTPUT FORMAT (JSON)
            ============================================================
            Return a single JSON object with the structure:

            {{
            "<cluster_name>": {{
                "phrases": [
                "...",
                "..."
                ],
                "description": "One-sentence explanation"
            }},
            ...
            }}

            ============================================================
            NOW CLUSTER THE FOLLOWING PHRASES:
            {phrase_list}

            Return ONLY the JSON object. No commentary.
            """