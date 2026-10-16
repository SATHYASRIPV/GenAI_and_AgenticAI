#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv
from agent import AIDataOrchestrator

load_dotenv()

def main():
    orchestrator = AIDataOrchestrator()
    
    if len(sys.argv) > 1:
        # Use command-line argument
        request = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("\n🤖 AI Tool Orchestrator")
        print("─" * 60)
        print("Examples:")
        print("  - Parse data/sample.csv and show summary")
        print("  - Load data/sales.csv, filter by region='North', calculate mean and sum")
        print("  - Convert data/file.json to CSV format")
        print("─" * 60)
        request = input("\nEnter your request: ").strip()
    
    orchestrator.orchestrate(request)

if __name__ == "__main__":
    main()