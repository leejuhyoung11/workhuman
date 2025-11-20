"""Router-enabled Workflow: task에 따라 다른 agent로 라우팅"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class WorkflowState(TypedDict):
    """State for Router workflow"""
    nom_id: int
    awards_text: str
    history_text: str
    task_type: str  # "extract", "judge", "explain"
    extracted_patterns: str
    prediction: str
    explanation: str
    final_output: str


class RouterWorkflow:
    """Router-enabled: task에 따라 다른 agent로 라우팅"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", anthropic_api_key: str = None):
        if anthropic_api_key:
            self.llm = ChatAnthropic(model=model_name, temperature=0, anthropic_api_key=anthropic_api_key)
        else:
            self.llm = ChatAnthropic(model=model_name, temperature=0)
        
        # Pattern Extractor Agent
        self.extractor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Pattern Extractor Agent.
Your role is to extract behavioral patterns, signals, and indicators from employee data."""),
            ("human", """Extract patterns from:

Awards:
{awards_text}

History:
{history_text}

Provide structured pattern extraction.""")
        ])
        
        # Judge Agent
        self.judge_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Judge Agent.
Your role is to make promotion predictions based on extracted patterns."""),
            ("human", """Extracted Patterns:
{extracted_patterns}

Make prediction: HIGH/MEDIUM/LOW with reasoning.""")
        ])
        
        # Explanation Agent
        self.explainer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Explanation Agent.
Your role is to provide clear, detailed explanations of predictions."""),
            ("human", """Prediction: {prediction}
Patterns: {extracted_patterns}

Provide detailed explanation for stakeholders.""")
        ])
    
    def extract_patterns(self, state: WorkflowState) -> WorkflowState:
        result = (self.extractor_prompt | self.llm).invoke({
            "awards_text": state["awards_text"],
            "history_text": state["history_text"]
        })
        state["extracted_patterns"] = result.content
        state["task_type"] = "extract"
        return state
    
    def judge_promotion(self, state: WorkflowState) -> WorkflowState:
        result = (self.judge_prompt | self.llm).invoke({
            "extracted_patterns": state["extracted_patterns"]
        })
        state["prediction"] = result.content
        state["task_type"] = "judge"
        return state
    
    def explain_result(self, state: WorkflowState) -> WorkflowState:
        result = (self.explainer_prompt | self.llm).invoke({
            "prediction": state["prediction"],
            "extracted_patterns": state["extracted_patterns"]
        })
        state["explanation"] = result.content
        state["final_output"] = f"Prediction: {state['prediction']}\n\nExplanation: {result.content}"
        state["task_type"] = "explain"
        return state
    
    def build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        workflow.add_node("extract_patterns", self.extract_patterns)
        workflow.add_node("judge_promotion", self.judge_promotion)
        workflow.add_node("explain_result", self.explain_result)
        
        workflow.set_entry_point("extract_patterns")
        
        # Routing logic: extract → judge → explain
        workflow.add_edge("extract_patterns", "judge_promotion")
        workflow.add_edge("judge_promotion", "explain_result")
        workflow.add_edge("explain_result", END)
        
        return workflow.compile()

