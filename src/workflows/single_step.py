"""Single-step Workflow: 모든 정보를 한 번에 LLM에 입력하여 판단"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class WorkflowState(TypedDict):
    """State for Single-step workflow"""
    nom_id: int
    awards_text: str
    history_text: str
    final_prediction: str
    reasoning: str


class SingleStepWorkflow:
    """Single-step: 모든 정보를 한 번에 LLM에 입력하여 판단"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", anthropic_api_key: str = None):
        if anthropic_api_key:
            self.llm = ChatAnthropic(model=model_name, temperature=0, anthropic_api_key=anthropic_api_key)
        else:
            self.llm = ChatAnthropic(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR analyst specializing in predicting employee promotions to VP level or higher positions.

Your task is to analyze employee award messages and career history to predict promotion potential.

Consider:
1. Leadership signals in award messages
2. Performance indicators
3. Career progression patterns
4. Manager relationships
5. Role changes and tenure

Provide a clear prediction (HIGH/MEDIUM/LOW) and detailed reasoning."""),
            ("human", """Employee ID: {nom_id}

Award Messages:
{awards_text}

Career History:
{history_text}

Based on this information, predict the likelihood of this employee being promoted to VP or higher. Provide:
1. Prediction: HIGH/MEDIUM/LOW
2. Detailed reasoning""")
        ])
    
    def analyze(self, state: WorkflowState) -> WorkflowState:
        """Single-step analysis"""
        print("[Single-step] Analyzing employee data...")
        chain = self.prompt | self.llm
        
        result = chain.invoke({
            "nom_id": state["nom_id"],
            "awards_text": state["awards_text"],
            "history_text": state["history_text"]
        })
        
        print("[Single-step] Analysis complete!")
        state["final_prediction"] = result.content
        state["reasoning"] = result.content
        return state
    
    def build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        workflow.add_node("analyze", self.analyze)
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", END)
        return workflow.compile()

