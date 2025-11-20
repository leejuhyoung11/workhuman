"""No-router Workflow: 모든 입력을 하나의 workflow가 처리"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class WorkflowState(TypedDict):
    """State for No-router workflow"""
    nom_id: int
    awards_text: str
    history_text: str
    extracted_patterns: str
    prediction: str
    explanation: str
    final_output: str


class NoRouterWorkflow:
    """No-router: 모든 입력을 하나의 workflow가 처리"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", anthropic_api_key: str = None):
        if anthropic_api_key:
            self.llm = ChatAnthropic(model=model_name, temperature=0, anthropic_api_key=anthropic_api_key)
        else:
            self.llm = ChatAnthropic(model=model_name, temperature=0)
        
        self.unified_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a comprehensive HR analysis system.
You handle all tasks: pattern extraction, prediction, and explanation in one unified process."""),
            ("human", """Employee ID: {nom_id}

Award Messages:
{awards_text}

Career History:
{history_text}

Perform complete analysis:
1. Extract key patterns
2. Predict promotion likelihood (HIGH/MEDIUM/LOW)
3. Provide detailed explanation

Provide all three outputs in a structured format.""")
        ])
    
    def unified_analysis(self, state: WorkflowState) -> WorkflowState:
        result = (self.unified_prompt | self.llm).invoke({
            "nom_id": state["nom_id"],
            "awards_text": state["awards_text"],
            "history_text": state["history_text"]
        })
        state["final_output"] = result.content
        # Parse output to populate other fields (simplified)
        state["extracted_patterns"] = result.content
        state["prediction"] = result.content
        state["explanation"] = result.content
        return state
    
    def build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        workflow.add_node("unified_analysis", self.unified_analysis)
        workflow.set_entry_point("unified_analysis")
        workflow.add_edge("unified_analysis", END)
        return workflow.compile()

