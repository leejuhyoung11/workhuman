"""Main entry point for VP promotion prediction experiment"""

import os, json
import sys
import pandas as pd
from dotenv import load_dotenv
from src.data_preprocessor import aggregate_employee_data, split_train_data
from src.data_loader import DataLoader
# from src.workflows.prompt_chaining import PromptChainingWorkflow

from sklearn.model_selection import train_test_split

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()




def main():
    """Main function"""
    
    
    # Check API key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY not found!")
        print("Please set it in .env file")
        return
    
    # Preprocess
    print("="*60)
    print("Data Preprocessing")
    print("="*60)
    employee_list = aggregate_employee_data(data_dir="data")
    
    
    print("employe len " + str(len(employee_list)))

    # print("\nSample of train data (JSON):")
    # print(json.dumps(employee_list[0], indent=2, ensure_ascii=False))

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=0,
        anthropic_api_key=anthropic_api_key
    )

    # workflow = PromptChainingWorkflow(llm)

    # result = workflow.run(train_list)

    # print(result["final_prediction"])
    # print(result["final_reasoning"])


    
if __name__ == "__main__":
    main()
