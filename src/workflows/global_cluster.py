from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from src.models.provider_factory import LLMProviderFactory

from utils.utils import parse_json_from_llm, save_final_result

class GlobalCluster:

    def __init__(self, provider, model, temperature, api_key):
        self.llm = LLMProviderFactory.create(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=4000
        )
    
    def dedupligate_signals(self, signal_list, is_vp):
        
        prompt = self._build_deduplicate_prompt(signal_list)
        raw = self.llm.call(prompt)
        print(" response:", raw)

        try:
            parsed_json =  parse_json_from_llm(raw)
        except Exception as e:
            print("JSON PARSE ERROR:", e)
            parsed_json = {}

        save_final_result(parsed_json, is_vp)

        return 




    def _build_deduplicate_prompt(self, signal_list):
        return f"""
            You will perform SEMANTIC DEDUPLICATION on a list of behavior cluster names.

            IMPORTANT:
            - These cluster names were already created by a previous clustering process.
            - Each name already represents a meaningful behavioral theme.
            - Your task is NOT to re-cluster raw phrases.
            - Your task is NOT to reinterpret employee-level behavior.
            - You must ONLY merge cluster names that clearly represent the SAME meaning.

            Do NOT:
            - generate new behavioral themes
            - split an existing cluster name into multiple new themes
            - infer seniority, rank, job level, or role type
            - reinterpret underlying meanings that were not in the names
            - create abstract or overly broad new categories

            You are strictly doing *semantic name consolidation*.

            ===============================================================
            STEP 1 — Understand the Input
            ===============================================================
            You are given a list of cluster names such as:
            - "Cross-functional Leadership"
            - "Leadership Across Teams"
            - "Cross-Team Influence"
            - "Team Coordination Leadership"

            These names may differ in wording but express similar concepts.
            Your job is to identify which names should be merged together.

            ===============================================================
            STEP 2 — Semantic Deduplication Rules
            ===============================================================
            For any set of names that express the same behavioral idea:
            1. Group them together.
            2. Select ONE canonical (representative) name.
            - Must be the clearest and simplest expression of the theme.
            - Should NOT be more abstract or broader than the originals.
            3. Store all original names under an "aliases" list.

            Important rules:
            - Merge ONLY if the meanings clearly match.
            - If a name is unique, keep it as its own canonical cluster.
            - Do NOT force a name into a group if the match is weak.
            - If two names overlap partially but are not conceptually identical, keep them separate.

            ===============================================================
            STEP 3 — Provide a Canonical Summary
            ===============================================================
            For each canonical cluster:
            Write a short 1–2 sentence summary explaining the core behavioral concept.
            This summary should:
            - describe the underlying competency
            - be neutral (no seniority inference)
            - reflect only the meaning shared by the aliases

            ===============================================================
            STEP 4 — Output Format (JSON Only)
            ===============================================================
            Return a single JSON object in the format:

            {{
            "<canonical_name>": {{
                "aliases": ["name1", "name2", "name3"],
                "summary": "1–2 sentence explanation of the shared meaning."
            }},
            ...
            }}

            Do NOT output anything outside of the JSON object.

            ===============================================================
            NOW DEDUPLICATE THE FOLLOWING CLUSTER NAMES:
            ===============================================================
            {signal_list}    
            """

    
    