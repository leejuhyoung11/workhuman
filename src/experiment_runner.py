"""Main experiment runner for comparing LangGraph workflows"""

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd

from src.data_loader import DataLoader
from src.evaluator import WorkflowEvaluator
from src.workflows.experiment1 import SingleStepWorkflow, PromptChainingWorkflow
from src.workflows.experiment2 import PromptChainingWorkflow as Exp2PromptChaining, MultiAgentWorkflow
from src.workflows.experiment3 import SequentialWorkflow, ParallelWorkflow
from src.workflows.experiment4 import NoRouterWorkflow, RouterWorkflow


load_dotenv()


class ExperimentRunner:
    """Run and compare LangGraph workflow experiments"""
    
    def __init__(self, data_dir: str = "data", sample_size: int = 10):
        self.data_loader = DataLoader(data_dir)
        self.evaluator = WorkflowEvaluator()
        self.sample_size = sample_size
        self.results = {}
    
    def prepare_test_data(self) -> List[Dict]:
        """Prepare test dataset with control and treatment samples"""
        control_df = self.data_loader.load_control()
        treatment_df = self.data_loader.load_treatment()
        
        # Sample employees
        control_employees = control_df['nom_id'].unique()[:self.sample_size]
        treatment_employees = treatment_df['nom_id'].unique()[:self.sample_size]
        
        test_data = []
        
        # Add control samples
        for nom_id in control_employees:
            try:
                data = self.data_loader.aggregate_employee_data(nom_id, "control")
                # Format for workflow input
                history_text = self._format_history(data['history'])
                test_data.append({
                    'nom_id': nom_id,
                    'awards_text': data['awards'],
                    'history_text': history_text,
                    'ground_truth': False
                })
            except Exception as e:
                print(f"Error processing control employee {nom_id}: {e}")
                continue
        
        # Add treatment samples
        for nom_id in treatment_employees:
            try:
                data = self.data_loader.aggregate_employee_data(nom_id, "treatment")
                history_text = self._format_history(data['history'])
                test_data.append({
                    'nom_id': nom_id,
                    'awards_text': data['awards'],
                    'history_text': history_text,
                    'ground_truth': True
                })
            except Exception as e:
                print(f"Error processing treatment employee {nom_id}: {e}")
                continue
        
        return test_data
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format history data into readable text"""
        if not history:
            return "No history data available"
        
        lines = []
        for entry in history:
            line = f"Role: {entry['job_title']}"
            if entry['manager_id']:
                line += f" | Manager ID: {entry['manager_id']}"
            if entry['start_date']:
                line += f" | Start: {entry['start_date']}"
            if entry['end_date']:
                line += f" | End: {entry['end_date']}"
            if entry['tenure_days']:
                line += f" | Tenure: {entry['tenure_days']} days"
            lines.append(line)
        
        return "\n".join(lines)
    
    def run_workflow(self, workflow, test_data: List[Dict]) -> List[Dict]:
        """Run a workflow on test data"""
        results = []
        
        for i, data in enumerate(test_data):
            print(f"Processing employee {i+1}/{len(test_data)}: {data['nom_id']}")
            
            try:
                # Prepare initial state
                initial_state = {
                    'nom_id': data['nom_id'],
                    'awards_text': data['awards_text'],
                    'history_text': data['history_text']
                }
                
                # Run workflow
                start_time = time.time()
                result = workflow.invoke(initial_state)
                elapsed_time = time.time() - start_time
                
                # Extract prediction (try multiple possible keys)
                prediction = (
                    result.get('final_prediction') or 
                    result.get('final_judgment') or 
                    result.get('final_output') or 
                    result.get('prediction') or 
                    ''
                )
                
                results.append({
                    'nom_id': data['nom_id'],
                    'prediction': prediction,
                    'ground_truth': data['ground_truth'],
                    'elapsed_time': elapsed_time,
                    'full_result': result
                })
                
            except Exception as e:
                print(f"Error running workflow for employee {data['nom_id']}: {e}")
                results.append({
                    'nom_id': data['nom_id'],
                    'prediction': 'ERROR',
                    'ground_truth': data['ground_truth'],
                    'elapsed_time': 0,
                    'error': str(e)
                })
        
        return results
    
    def run_experiment1(self, test_data: List[Dict]) -> Dict:
        """Experiment 1: Single-step vs Prompt Chaining"""
        print("\n" + "="*60)
        print("Experiment 1: Single-step LLM vs Prompt Chaining")
        print("="*60)
        
        # Single-step workflow
        print("\nRunning Single-step workflow...")
        single_step = SingleStepWorkflow().build_graph()
        single_step_results = self.run_workflow(single_step, test_data)
        
        # Prompt chaining workflow
        print("\nRunning Prompt Chaining workflow...")
        prompt_chaining = PromptChainingWorkflow().build_graph()
        chaining_results = self.run_workflow(prompt_chaining, test_data)
        
        # Evaluate
        single_step_eval = self.evaluator.evaluate_workflow(
            "Single-step",
            [r['prediction'] for r in single_step_results],
            [r['ground_truth'] for r in single_step_results]
        )
        
        chaining_eval = self.evaluator.evaluate_workflow(
            "Prompt Chaining",
            [r['prediction'] for r in chaining_results],
            [r['ground_truth'] for r in chaining_results]
        )
        
        comparison = self.evaluator.compare_workflows(
            "Single-step", single_step_eval,
            "Prompt Chaining", chaining_eval
        )
        
        return {
            'experiment': 'Experiment 1',
            'results': [single_step_eval, chaining_eval],
            'comparison': comparison,
            'raw_results': {
                'single_step': single_step_results,
                'chaining': chaining_results
            }
        }
    
    def run_experiment2(self, test_data: List[Dict]) -> Dict:
        """Experiment 2: Prompt Chaining vs Multi-agent"""
        print("\n" + "="*60)
        print("Experiment 2: Prompt Chaining vs Multi-agent Workflow")
        print("="*60)
        
        # Prompt chaining (from exp2)
        print("\nRunning Prompt Chaining workflow...")
        prompt_chaining = Exp2PromptChaining().build_graph()
        chaining_results = self.run_workflow(prompt_chaining, test_data)
        
        # Multi-agent workflow
        print("\nRunning Multi-agent workflow...")
        multi_agent = MultiAgentWorkflow().build_graph()
        multi_agent_results = self.run_workflow(multi_agent, test_data)
        
        # Evaluate
        chaining_eval = self.evaluator.evaluate_workflow(
            "Prompt Chaining",
            [r['prediction'] for r in chaining_results],
            [r['ground_truth'] for r in chaining_results]
        )
        
        multi_agent_eval = self.evaluator.evaluate_workflow(
            "Multi-agent",
            [r['prediction'] for r in multi_agent_results],
            [r['ground_truth'] for r in multi_agent_results]
        )
        
        comparison = self.evaluator.compare_workflows(
            "Prompt Chaining", chaining_eval,
            "Multi-agent", multi_agent_eval
        )
        
        return {
            'experiment': 'Experiment 2',
            'results': [chaining_eval, multi_agent_eval],
            'comparison': comparison,
            'raw_results': {
                'chaining': chaining_results,
                'multi_agent': multi_agent_results
            }
        }
    
    def run_experiment3(self, test_data: List[Dict]) -> Dict:
        """Experiment 3: Sequential vs Parallel"""
        print("\n" + "="*60)
        print("Experiment 3: Sequential vs Parallel Workflow")
        print("="*60)
        
        # Sequential workflow
        print("\nRunning Sequential workflow...")
        sequential = SequentialWorkflow().build_graph()
        sequential_results = self.run_workflow(sequential, test_data)
        
        # Parallel workflow
        print("\nRunning Parallel workflow...")
        parallel = ParallelWorkflow().build_graph()
        parallel_results = self.run_workflow(parallel, test_data)
        
        # Evaluate
        sequential_eval = self.evaluator.evaluate_workflow(
            "Sequential",
            [r['prediction'] for r in sequential_results],
            [r['ground_truth'] for r in sequential_results]
        )
        
        parallel_eval = self.evaluator.evaluate_workflow(
            "Parallel",
            [r['prediction'] for r in parallel_results],
            [r['ground_truth'] for r in parallel_results]
        )
        
        comparison = self.evaluator.compare_workflows(
            "Sequential", sequential_eval,
            "Parallel", parallel_eval
        )
        
        return {
            'experiment': 'Experiment 3',
            'results': [sequential_eval, parallel_eval],
            'comparison': comparison,
            'raw_results': {
                'sequential': sequential_results,
                'parallel': parallel_results
            }
        }
    
    def run_experiment4(self, test_data: List[Dict]) -> Dict:
        """Experiment 4: Router-enabled vs No-router"""
        print("\n" + "="*60)
        print("Experiment 4: Router-enabled vs No-router Workflow")
        print("="*60)
        
        # No-router workflow
        print("\nRunning No-router workflow...")
        no_router = NoRouterWorkflow().build_graph()
        no_router_results = self.run_workflow(no_router, test_data)
        
        # Router workflow
        print("\nRunning Router-enabled workflow...")
        router = RouterWorkflow().build_graph()
        router_results = self.run_workflow(router, test_data)
        
        # Evaluate
        no_router_eval = self.evaluator.evaluate_workflow(
            "No-router",
            [r['prediction'] for r in no_router_results],
            [r['ground_truth'] for r in no_router_results]
        )
        
        router_eval = self.evaluator.evaluate_workflow(
            "Router-enabled",
            [r['prediction'] for r in router_results],
            [r['ground_truth'] for r in router_results]
        )
        
        comparison = self.evaluator.compare_workflows(
            "No-router", no_router_eval,
            "Router-enabled", router_eval
        )
        
        return {
            'experiment': 'Experiment 4',
            'results': [no_router_eval, router_eval],
            'comparison': comparison,
            'raw_results': {
                'no_router': no_router_results,
                'router': router_results
            }
        }
    
    def run_all_experiments(self) -> Dict:
        """Run all experiments"""
        print("Preparing test data...")
        test_data = self.prepare_test_data()
        print(f"Prepared {len(test_data)} test cases")
        
        all_results = {}
        
        # Run each experiment
        all_results['experiment1'] = self.run_experiment1(test_data)
        all_results['experiment2'] = self.run_experiment2(test_data)
        all_results['experiment3'] = self.run_experiment3(test_data)
        all_results['experiment4'] = self.run_experiment4(test_data)
        
        self.results = all_results
        return all_results
    
    def generate_final_report(self) -> str:
        """Generate final comprehensive report"""
        report = "\n" + "="*80 + "\n"
        report += "FINAL EXPERIMENT REPORT\n"
        report += "="*80 + "\n\n"
        
        for exp_key, exp_data in self.results.items():
            report += self.evaluator.generate_report(
                exp_data['experiment'],
                exp_data['results']
            )
            report += "\nComparison:\n"
            report += exp_data['comparison'].to_string(index=False)
            report += "\n\n" + "-"*80 + "\n\n"
        
        return report

