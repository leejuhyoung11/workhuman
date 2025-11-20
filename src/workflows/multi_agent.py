#############Multi-agent Workflow##############

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class WorkflowState(TypedDict):
    """State for Multi-agent workflow"""
    nom_id: int
    awards_text: str
    history_text: str
    award_extraction: str
    history_extraction: str
    aggregated_evidence: str
    final_judgment: str
    reasoning: str


class MultiAgentWorkflow:
    """Multi-agent: 역할 분리된 에이전트들"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", anthropic_api_key: str = None):
        if anthropic_api_key:
            self.llm = ChatAnthropic(model=model_name, temperature=0, anthropic_api_key=anthropic_api_key)
        else:
            self.llm = ChatAnthropic(model=model_name, temperature=0)
        
        # Award Extractor Agent
        self.award_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Award Extractor Agent specializing in identifying:
            - Leadership indicators
            - Performance achievements
            - Collaboration skills
            - Strategic contributions
            from award messages."""),
            ("human", "Extract signals from:\n{awards_text}")
        ])
        
        # History Pattern Agent
        self.history_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a History Pattern Agent specializing in identifying:
            - Career progression trajectories
            - Role change patterns
            - Manager relationship dynamics
            - Tenure and stability indicators
            from career history."""),
            ("human", "Analyze patterns in:\n{history_text}")
        ])
        
        # Evidence Aggregator Agent
        self.aggregator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Evidence Aggregator Agent. Your role is to:
            - Combine insights from multiple sources
            - Identify correlations and contradictions
            - Create a unified evidence base
            - Prepare for final judgment"""),
            ("human", """Award Analysis:
            {award_extraction}

            History Analysis:
            {history_extraction}

            Aggregate and synthesize this evidence.""")
        ])
        
        # Judge Agent
        self.judge_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Judge Agent with expertise in VP-level promotion decisions.
            You make final predictions based on aggregated evidence."""),
            ("human", """Aggregated Evidence:
            {aggregated_evidence}

            Make your final judgment:
            1. Prediction: HIGH/MEDIUM/LOW
            2. Detailed reasoning""")
        ])
    
    def award_extractor(self, state: WorkflowState) -> WorkflowState:
        print("[Multi-agent] Agent 1/4: Award Extractor working...")
        result = (self.award_agent_prompt | self.llm).invoke({"awards_text": state["awards_text"]})
        print("[Multi-agent] Agent 1/4: Complete!")
        state["award_extraction"] = result.content
        return state
    
    def history_pattern_agent(self, state: WorkflowState) -> WorkflowState:
        print("[Multi-agent] Agent 2/4: History Pattern Agent working...")
        result = (self.history_agent_prompt | self.llm).invoke({"history_text": state["history_text"]})
        print("[Multi-agent] Agent 2/4: Complete!")
        state["history_extraction"] = result.content
        return state
    
    def evidence_aggregator(self, state: WorkflowState) -> WorkflowState:
        print("[Multi-agent] Agent 3/4: Evidence Aggregator working...")
        result = (self.aggregator_prompt | self.llm).invoke({
            "award_extraction": state["award_extraction"],
            "history_extraction": state["history_extraction"]
        })
        print("[Multi-agent] Agent 3/4: Complete!")
        state["aggregated_evidence"] = result.content
        return state
    
    def judge(self, state: WorkflowState) -> WorkflowState:
        print("[Multi-agent] Agent 4/4: Judge Agent making final decision...")
        result = (self.judge_prompt | self.llm).invoke({"aggregated_evidence": state["aggregated_evidence"]})
        print("[Multi-agent] Agent 4/4: Complete!")
        state["final_judgment"] = result.content
        state["reasoning"] = result.content
        return state
    
    def build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        workflow.add_node("award_extractor", self.award_extractor)
        workflow.add_node("history_pattern_agent", self.history_pattern_agent)
        workflow.add_node("evidence_aggregator", self.evidence_aggregator)
        workflow.add_node("judge", self.judge)
        
        workflow.set_entry_point("award_extractor")
        workflow.add_edge("award_extractor", "history_pattern_agent")
        workflow.add_edge("history_pattern_agent", "evidence_aggregator")
        workflow.add_edge("evidence_aggregator", "judge")
        workflow.add_edge("judge", END)
        
        return workflow.compile()

