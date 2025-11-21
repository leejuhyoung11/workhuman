from typing import List, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic

from utils.utils import chunk_awards, parse_json_from_llm, save_employee_keywords

class PromptChainingWorkflow:

    def __init__(self, model, temperature, api_key):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    # -------------------------
    # STEP 1: Award Chunk Summaries
    # -------------------------
    def extract_raw_keywords(self, employee) -> str:
        rec_id = employee.get("rec_id")
        awards = employee.get("awards")

        award_chunks = chunk_awards(rec_id=rec_id, awards_list=awards)

        def process_chunk(chunk_text):
            llm = Anthropic(api_key=self.api_key)

            prompt = self._build_extracting_signal_prompt(chunk_text)
            
            response = llm.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

            raw = response.content[0].text
            print(" response:", raw)

            try:
                return parse_json_from_llm(raw)
            except Exception as e:
                print("JSON PARSE ERROR:", e)
                return {} 

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(process_chunk, award_chunks[0:2])

        all_results = {}
        keyword_set = set()

        for r in results:
            # skip invalid or empty
            if not r or not isinstance(r, dict):
                continue

            for award_index, sentence_dict in r.items():
                award_index = str(award_index)

                if not isinstance(sentence_dict, dict):
                    print(f"[WARN] Invalid sentence_dict for award {award_index}: {sentence_dict}")
                    continue

                all_results[award_index] = sentence_dict

        save_employee_keywords(rec_id=rec_id, results=all_results)


        return rec_id, all_results




    # -------------------------
    # STEP 2: Employee-Level Summary (Merge all chunk summaries)
    # -------------------------
    def summarize_employee(self, chunk_summaries: List[str]) -> str:
        """Merge chunk summaries into one employee-level summary."""
        prompt = self._build_employee_summary_prompt(chunk_summaries)
        response = self.llm.invoke(prompt)
        return response.content

    # -------------------------
    # STEP 3: VP vs Non-VP Pattern Comparison
    # -------------------------
    def compare_groups(self, vp_summaries: List[str], non_vp_summaries: List[str]) -> str:
        """Compare VP vs Non-VP summaries and extract pattern differences."""
        prompt = self._build_group_comparison_prompt(vp_summaries, non_vp_summaries)
        response = self.llm.invoke(prompt)
        return response.content

    # ========================================================
    # PROMPT BUILDERS
    # ========================================================

    # STEP 1 Prompt ------------------------------------------
    def _build_extracting_signal_prompt(self, chunk_text):
        return f"""
                You will analyze multiple awards from a single employee.

                Each award entry is provided on one line using the format:

                <award_index># <title> | <message>

                Where:
                - <award_index> is the integer before the "#".
                - <title> appears before the "|".
                - <message> appears after the "|", and may contain multiple sentences.

                --------------------------------------------------------
                YOUR TASKS
                --------------------------------------------------------

                1. **award_index**
                - Use the number before the "#" as the award_index.
                - Do NOT generate new numbers or reorder awards.

                2. **Title processing (sentence_index = 0)**
                - Titles may contain meaningful competency keywords.
                - If the title contains a clear behavioral/competency concept
                    ("leadership", "innovation", "collaboration", "excellence",
                        "strategic", "client", "mentorship", "execution", "impact")
                    then extract exactly one signal.
                - If the title is non-informative ("Award", "Recognition", etc.),
                    omit the title entirely from the output.
                - If you extract a signal from the title, it must be stored under:
                    "0": ["signal"]

                3. **Message sentence processing**
                - Split the message into sentences.
                - Assign sentence_index starting from 1 based on the ORIGINAL order.
                - For each message sentence:
                    • Extract 0–2 behavioral signals.
                    • A signal is a short (1–3 word) competency descriptor.
                    • If no meaningful signal exists → OMIT the sentence entirely.

                IMPORTANT RULE ABOUT SENTENCE INDEXING:
                - Preserve the original sentence numbering.
                - Do NOT renumber.
                - Do NOT collapse missing indices.
                - Example:
                    A message has 5 sentences.
                    Only sentence 2 and 4 contain signals.
                    Output must include ONLY:
                    "2": [...],
                    "4": [...]
                    and must NOT include 1, 3, or 5.

                4. **Output Format**
                - Output a SINGLE JSON object.
                - Keys are award_index (as strings).
                - Each award_index maps to an object whose keys are the 
                    sentence_index values (as strings), and whose values 
                    are lists of extracted signal strings.
                Example:
                {{
                    "0": {{
                    "0": ["leadership"],
                    "2": ["team coordination"],
                    "4": ["strategic direction"]
                    }},
                    "1": {{
                    "1": ["stakeholder communication"]
                    }}
                }}

                5. Do NOT output sentence text.
                6. Do NOT output commentary, explanation, or any extra text.
                7. Return JSON only.
                --------------------------------------------------------
                FEW-SHOT EXAMPLES
                --------------------------------------------------------

                Example Input:
                0# Outstanding Team Collaboration | John coordinated three teams. He resolved conflicts quickly.
                1# Client Excellence Award | She built trust with a difficult client. Her communication kept stakeholders confident.

                Example Output:
                {{
                "0": {{
                    "0": ["collaboration"],
                    "1": ["cross-team coordination"],
                    "2": ["conflict resolution"]
                }},
                "1": {{
                    "1": ["client relationship building"],
                    "2": ["stakeholder communication"]
                }}
                }}

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
                - Title “Recognition Award” → NOT meaningful → omitted.
                - Message sentence 1: no meaningful signal → omitted.
                - Message sentence 2: contains “dedication” → included as "2": ["dedication"].


                --------------------------------------------------------
                NOW PROCESS THE FOLLOWING AWARDS:
                {chunk_text}

                Return ONLY a JSON object.
            """

    # STEP 3 Prompt ------------------------------------------
    def _build_group_comparison_prompt(self, vp_summaries: List[str], non_vp_summaries: List[str]) -> str:
        vp_text = "\n".join([f"- {s}" for s in vp_summaries])
        non_vp_text = "\n".join([f"- {s}" for s in non_vp_summaries])

        return f"""
            You are comparing two groups of employees based on their award summaries.

            Group A = VP (promoted employees)  
            Group B = Non-VP (not promoted employees)

            Your task is to identify **true behavioral and linguistic differences** that appear consistently in VP summaries but not in Non-VP summaries.

            ### Requirements:
            - Base all conclusions **only** on the provided summaries. 
            - Identify strong differentiating themes.
            - Identify weak or noisy signals (appear in both groups).
            - Identify themes that are common baseline behaviors.
            - DO NOT speculate beyond the given text.
            - Output must be structured and clear.

            ### VP Employee Summaries:
            {vp_text}

            ### Non-VP Employee Summaries:
            {non_vp_text}

            ### Output:
            Provide a structured comparison:

            1. **Top Differentiators** (very strong signals more common in VP group)
            2. **Baseline Behaviors** (similar levels in both groups)
            3. **Noisy / Non-predictive Signals** (random or inconsistent)
            4. **Explanation** of why these differences matter
            """