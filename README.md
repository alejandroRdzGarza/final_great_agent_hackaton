# Great Agent AI Hackathon 2025 Submissions

This repository contains our team's submissions for the Great Agent AI Hackathon 2025.

## 📁 Projects

### Track B: Multi-Agent Insurance Claims Processing
**Location:** `/Track_B/`
# Insurance Company Multi-Agent System

## 🏢 Overview

This project simulates an insurance company workflow where AI agents take on different roles to process insurance claims. Each agent has specialized expertise, and through discussion and analysis, they collaboratively reach decisions on claim outcomes.

## 🔍 Key Features: Observability & Interpretability

### LangSmith Integration for Full Transparency
- **Conversation Thread Monitoring**: Every agent discussion is tracked and visible in LangSmith
- **Decision Tracing**: Follow the complete reasoning chain from claim intake to final verdict
- **Performance Metrics**: Monitor agent response times, token usage, and decision quality
- **Audit Trails**: Complete history of all agent interactions and tool usage

### Justification-First Agent Design
- **Evidence-Based Reasoning**: Agents must cite specific policy clauses, calculations, or research findings
- **Confidence Scoring**: Every assessment includes a confidence level (0-100%)
- **Alternative Scenarios**: Agents consider and explain why they rejected other possible outcomes
- **Transparent Research**: Valyu search queries and results are logged and attributable

## 👥 Agent Roles & Their Justification Requirements

- **SIU Investigator**: Must provide evidence for fraud suspicions, cite search results, and quantify risk levels
- **Claims Adjuster**: Must show damage calculations, policy references, and valuation methodology  
- **Transparency Auditor**: Must verify compliance with specific policy sections and process steps
- **Claims Manager**: Must synthesize all inputs and explain the final decision with weighted reasoning

### Track C: LLM Security & Red Teaming  
**Location:** `/Track_C/`

A framework focused on testing and breaking LLM security through various attack techniques and vulnerability analysis.

## 🚀 Quick Start

```bash
# Track B - Insurance Agent System
cd Track_B
conda env create -f environment.yaml
conda activate [env-name]
python workflow.py

# Track C - LLM Security Testing
cd Track_C  
pip install -r requirements.txt
# See Track_C/README.md for specific instructions