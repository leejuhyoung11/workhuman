"""Parallel Workflow: award와 history를 동시에 분석 → aggregator에서 통합"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class WorkflowState(TypedDict):
    """State for Parallel workflow"""
    nom_id: int
    awards_text: str
    history_text: str
    award_analysis: str
    history_analysis: str
    final_prediction: str
    reasoning: str


class ParallelWorkflow:
    """Parallel: award와 history를 동시에 분석 → aggregator에서 통합"""
    
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
        
        self.aggregator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an aggregator that combines parallel analyses.
Synthesize insights from both award and history analyses to make a unified prediction."""),
            ("human", """Award Analysis (parallel):
{award_analysis}

History Analysis (parallel):
{history_analysis}

Provide unified prediction: HIGH/MEDIUM/LOW with comprehensive reasoning.""")
        ])
    
    def analyze_awards_parallel(self, state: WorkflowState) -> WorkflowState:
        result = (self.award_prompt | self.llm).invoke({"awards_text": state["awards_text"]})
        state["award_analysis"] = result.content
        return state
    
    def analyze_history_parallel(self, state: WorkflowState) -> WorkflowState:
        result = (self.history_prompt | self.llm).invoke({"history_text": state["history_text"]})
        state["history_analysis"] = result.content
        return state
    
    def aggregate_and_predict(self, state: WorkflowState) -> WorkflowState:
        result = (self.aggregator_prompt | self.llm).invoke({
            "award_analysis": state["award_analysis"],
            "history_analysis": state["history_analysis"]
        })
        state["final_prediction"] = result.content
        state["reasoning"] = result.content
        return state
    
    def build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        workflow.add_node("analyze_awards_parallel", self.analyze_awards_parallel)
        workflow.add_node("analyze_history_parallel", self.analyze_history_parallel)
        workflow.add_node("aggregate_and_predict", self.aggregate_and_predict)
        
        workflow.set_entry_point("analyze_awards_parallel")
        # Sequential execution for now (true parallelism would require async/threading)
        workflow.add_edge("analyze_awards_parallel", "analyze_history_parallel")
        workflow.add_edge("analyze_history_parallel", "aggregate_and_predict")
        workflow.add_edge("aggregate_and_predict", END)
        
        return workflow.compile()

