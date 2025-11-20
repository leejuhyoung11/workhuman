from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Prompt Chaining Workflow Class
# -----------------------------
class PromptChainingWorkflow:

    def __init__(self, llm):
        """
        llm: ChatAnthropic / ChatGoogle / ChatOpenAI 
        """
        self.llm = llm
        self._build_prompts()

    # ----------------------------------------
    # Step 0 – Prompt Template Building
    # ----------------------------------------
    def _build_prompts(self):

        # 1) Award Parsing
        self.parse_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at interpreting employee award texts."),
            ("human", """Parse the following awards into structured signals.
                Each award contains: title, message, date.

                Awards JSON:
                {awards}

                Extract:
                1. Leadership behaviors
                2. Collaboration signals
                3. Execution ability
                4. Strategic impact

                Return a structured and concise analysis."""
            )
        ])

        # 2) Feature Extraction
        self.feature_prompt = ChatPromptTemplate.from_messages([
            ("system", "You analyze leadership potential indicators."),
            ("human", """Based on parsed award information below:

                {parsed_awards}

                Extract high-level features:
                - Leadership indicators
                - Influence patterns
                - Cross-team collaboration
                - Innovation or initiative
                - VP potential signals

                Return a JSON-style bullet summary."""
            )
        ])

        # 3) Scoring + Prediction
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", "You predict VP promotion likelihood from award-based features."),
            ("human", """Based on features:

                {extracted_features}

                Score the employee's VP readiness.

                Return:
                1. Score (HIGH / MEDIUM / LOW)
                2. Reasoning that justifies the score."""
            )
        ])

    # ----------------------------------------
    # Node 1 – Parse Awards
    # ----------------------------------------
    def parse_awards(self, state: WorkflowState) -> WorkflowState:
        chain = self.parse_prompt | self.llm
        result = chain.invoke({"awards": state["awards"]})
        state["parsed_awards"] = result.content
        return state

    # ----------------------------------------
    # Node 2 – Extract Features
    # ----------------------------------------
    def extract_features(self, state: WorkflowState) -> WorkflowState:
        chain = self.feature_prompt | self.llm
        result = chain.invoke({"parsed_awards": state["parsed_awards"]})
        state["extracted_features"] = result.content
        return state

    # ----------------------------------------
    # Node 3 – Score & Final Prediction
    # ----------------------------------------
    def score_candidate(self, state: WorkflowState) -> WorkflowState:
        chain = self.scoring_prompt | self.llm
        result = chain.invoke({"extracted_features": state["extracted_features"]})
        state["final_prediction"] = result.content
        state["final_reasoning"] = result.content
        return state

    # ----------------------------------------
    # Build LangGraph
    # ----------------------------------------
    def build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("parse_awards", self.parse_awards)
        graph.add_node("extract_features", self.extract_features)
        graph.add_node("score_candidate", self.score_candidate)

        graph.set_entry_point("parse_awards")
        graph.add_edge("parse_awards", "extract_features")
        graph.add_edge("extract_features", "score_candidate")
        graph.add_edge("score_candidate", END)

        return graph.compile()

    # ----------------------------------------
    # Public Run Method
    # ----------------------------------------
    def run(self, employee_json: Dict[str, Any]) -> Dict[str, Any]:
        graph = self.build_graph()

        # Generate Initial State 
        init_state: WorkflowState = {
            "nom_id": employee_json["nom_id"],
            "awards": employee_json["awards"],
            "parsed_awards": "",
            "extracted_features": "",
            "score": "",
            "final_prediction": "",
            "final_reasoning": "",
        }

        final_state = graph.invoke(init_state)
        return final_state