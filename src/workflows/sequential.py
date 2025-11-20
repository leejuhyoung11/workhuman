"""Sequential Workflow: award 분석 → history 분석 → 결합"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class WorkflowState(TypedDict):
    """State for Sequential workflow"""
    nom_id: int
    awards_text: str
    history_text: str
    award_analysis: str
    history_analysis: str
    final_prediction: str
    reasoning: str


class SequentialWorkflow:
    """Sequential: award 분석 → history 분석 → 결합"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", anthropic_api_key: str = None):
        if anthropic_api_key:
            self.llm = ChatAnthropic(model=model_name, temperature=0, anthropic_api_key=anthropic_api_key)
        else:
            self.llm = ChatAnthropic(model=model_name, temperature=0)
        
        self.award_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze award messages for promotion signals."),
            ("human", "Awards:\n{awards_text}")
        ])
        
        self.history_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze career history for progression patterns."),
            ("human", "History:\n{history_text}")
        ])
        
        self.combine_prompt = ChatPromptTemplate.from_messages([
            ("system", "Combine analyses and predict promotion likelihood."),
            ("human", """Award Analysis: {award_analysis}
History Analysis: {history_analysis}

Provide final prediction: HIGH/MEDIUM/LOW with reasoning.""")
        ])
    
    def analyze_awards(self, state: WorkflowState) -> WorkflowState:
        result = (self.award_prompt | self.llm).invoke({"awards_text": state["awards_text"]})
        state["award_analysis"] = result.content
        return state
    
    def analyze_history(self, state: WorkflowState) -> WorkflowState:
        result = (self.history_prompt | self.llm).invoke({"history_text": state["history_text"]})
        state["history_analysis"] = result.content
        return state
    
    def combine_and_predict(self, state: WorkflowState) -> WorkflowState:
        result = (self.combine_prompt | self.llm).invoke({
            "award_analysis": state["award_analysis"],
            "history_analysis": state["history_analysis"]
        })
        state["final_prediction"] = result.content
        state["reasoning"] = result.content
        return state
    
    def build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        workflow.add_node("analyze_awards", self.analyze_awards)
        workflow.add_node("analyze_history", self.analyze_history)
        workflow.add_node("combine_and_predict", self.combine_and_predict)
        
        workflow.set_entry_point("analyze_awards")
        workflow.add_edge("analyze_awards", "analyze_history")
        workflow.add_edge("analyze_history", "combine_and_predict")
        workflow.add_edge("combine_and_predict", END)
        
        return workflow.compile()

