import os
import re
import logging
import asyncio
from typing import List, Dict, TypedDict, Annotated, Tuple

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Local imports
from rag_pipeline import rag_pipeline
from llm_utils import call_llm_api

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Agent State Definition ---
class AgentState(TypedDict):
    query: str
    uo_block: str
    uo_id: str
    uo_name: str
    section_to_populate: str
    options: Dict[str, List[str]]
    messages: Annotated[list, add_messages]

# --- Helper function for content generation ---
def _extract_section_content(uo_block: str, section_name: str) -> str:
    """Helper to extract content of a specific section from a UO block."""
    pattern = re.compile(r"#### " + re.escape(section_name) + r"\n(.*?)(?=\n####|\n------------------------------------------------------------------------)", re.DOTALL)
    match = pattern.search(uo_block)
    if match:
        content = match.group(1).strip()
        return content if content and not content.startswith('(') else "(not specified)"
    return "(not specified)"

async def _generate_options(query: str, uo_id: str, uo_name: str, section: str, uo_block: str) -> Tuple[List[str], str]:
    """
    RAG 검색 결과에 따라 동적으로 프롬프트를 조정하고, 출처 정보 문자열을 함께 반환합니다.
    - 각 LLM에서 최상의 답변 하나씩을 생성하여 총 3가지 옵션을 반환합니다.
    """
    logger.info(f"Generating options for UO '{uo_id}' - Section '{section}' using multiple LLMs")
    input_context = _extract_section_content(uo_block, "Input")
    rag_query = f"Find the specific procedure or list of items for the '{section}' section of the unit operation '{uo_id}: {uo_name}' related to the experiment: {query}"

    logger.info(f"Refined RAG Query: {rag_query}")
    context_docs = rag_pipeline.retrieve_context(rag_query, k=3)
    rag_context = rag_pipeline.format_context_for_prompt(context_docs)
    attribution_str = ""

    if "No relevant context found" in rag_context or not rag_context.strip():
        logger.warning(f"No relevant SOPs found for '{section}' in '{uo_name}'. Falling back to general knowledge.")
        attribution_str = "[주의: 참고할 SOP가 없어 LLM의 자체 지식으로 생성됨]"
        base_user_prompt = f"""
- **Experiment Goal**: '{query}'
- **Unit Operation**: '{uo_id}: {uo_name}'
- **Section to Write**: '{section}'
- **Inputs**: '{input_context}'

Based on your general molecular biology knowledge, please write a plausible list or protocol for the '{section}'.
"""
    else:
        sources = list(set([doc.metadata.get('source', 'Unknown').split('/')[-1] for doc in context_docs]))
        attribution_str = f"[참고 SOP: {', '.join(sources)}]"
        base_user_prompt = f"""
- **Experiment Goal**: '{query}'
- **Unit Operation**: '{uo_id}: {uo_name}'
- **Section to Write**: '{section}'
- **Inputs**: '{input_context}'

--- **Relevant SOP Context** ---
{rag_context}
---

Your task is to write the content for the specified section using the provided SOP context.
"""

    # 사용할 LLM 모델 목록 정의
    models_to_use = ["biollama3", "mixtral", "llama3:70b"]

    # 각 모델별로 최적의 답변을 유도하기 위한 통합 프롬프트
    # DPO 학습 시 'concise'와 'detailed' 사이의 균형을 학습시키는 것을 목표로 합니다.
    unified_prompt = {
        "system": "You are a specialized scientific assistant. Your task is to generate a comprehensive and well-structured response for a specific section of a lab note, using the provided context. The response should be clear, detailed, and directly applicable to the experiment. Your answer MUST be only the list or method itself, without any extra conversation or explanation.",
        "user": base_user_prompt
    }

    tasks = [
        call_llm_api(
            system_prompt=unified_prompt["system"],
            user_prompt=unified_prompt["user"],
            model_name=model_name
        )
        for model_name in models_to_use
    ]

    generated_options = await asyncio.gather(*tasks)

    # 생성된 옵션들을 필터링하고 attribution 정보를 추가합니다.
    final_options = []
    model_titles = {
        "biollama3": "BioLLaMa3 (Default)",
        "mixtral": "Mixtral 8x7B (Efficient)",
        "llama3:70b": "Llama 3 70B (High-Quality)"
    }
    
    for i, opt in enumerate(generated_options):
        if opt and not opt.startswith("(LLM Error"):
            model_name = models_to_use[i]
            title = model_titles.get(model_name, model_name)
            
            # 각 모델의 출력을 구별하기 위해 제목을 추가합니다.
            formatted_option = f"--- {title}의 제안 ---\n\n{opt}"
            
            # 원본 SOP 출처 정보를 맨 뒤에 추가합니다.
            final_options.append(f"{formatted_option}\n\n{attribution_str}")
    
    return final_options, attribution_str
    
# --- Agent Nodes ---
def method_agent(state: AgentState) -> AgentState:
    logger.info(f"Method Agent: Generating content for {state['uo_id']}")
    options, attribution = asyncio.run(
        _generate_options(state['query'], state['uo_id'], state['uo_name'], 'Method', state['uo_block'])
    )
    state['options']['Method'] = [f"{attribution}\n\n{opt}" for opt in options]
    return state

def materials_agent(state: AgentState) -> AgentState:
    section = state['section_to_populate']
    logger.info(f"Materials Agent: Generating content for {state['uo_id']} - {section}")
    options, attribution = asyncio.run(
        _generate_options(state['query'], state['uo_id'], state['uo_name'], section, state['uo_block'])
    )
    state['options'][section] = [f"{attribution}\n\n{opt}" for opt in options]
    return state

def results_agent(state: AgentState) -> AgentState:
    section = state['section_to_populate']
    logger.info(f"Results Agent: Generating content for {state['uo_id']} - {section}")
    options, attribution = asyncio.run(
        _generate_options(state['query'], state['uo_id'], state['uo_name'], section, state['uo_block'])
    )
    state['options'][section] = [f"{attribution}\n\n{opt}" for opt in options]
    return state


# --- Routing Logic ---
def route_request(state: AgentState) -> str:
    logger.info(f"Router: Routing task for UO '{state['uo_id']}' - Section: '{state['section_to_populate']}'")
    section = state['section_to_populate']
    
    if section == "Method":
        return "method_agent"
    elif section in ["Input", "Reagent", "Consumables", "Equipment"]:
        return "materials_agent"
    elif section in ["Output", "Results & Discussions"]:
        return "results_agent"
    else:
        logger.warning(f"Router: No agent found for section '{section}'. Ending execution.")
        return END

# --- Graph Definition ---
def create_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("method_agent", method_agent)
    graph.add_node("materials_agent", materials_agent)
    graph.add_node("results_agent", results_agent)
    graph.set_conditional_entry_point(
        route_request,
        {
            "method_agent": "method_agent",
            "materials_agent": "materials_agent",
            "results_agent": "results_agent",
            END: END
        }
    )
    graph.add_edge("method_agent", END)
    graph.add_edge("materials_agent", END)
    graph.add_edge("results_agent", END)
    
    agent_graph = graph.compile()
    logger.info("Agent graph compiled successfully with conditional entry point.")
    return agent_graph

# --- Main execution function ---
def run_agent_team(query: str, uo_block: str, section: str) -> Dict:
    match = re.search(r"### \[(U[A-Z]{2,3}\d{3}) (.*)\]", uo_block)
    if not match:
        logger.error(f"Could not parse UO ID and Name from block.")
        return {}
        
    uo_id, uo_name = match.groups()

    initial_state = AgentState(
        query=query,
        uo_block=uo_block,
        uo_id=uo_id,
        uo_name=uo_name,
        section_to_populate=section,
        options={},
        messages=[]
    )
    
    graph = create_agent_graph()
    final_state = graph.invoke(initial_state)
    
    return {
        "uo_id": uo_id,
        "section": section,
        "options": final_state.get('options', {}).get(section, [])
    }

if __name__ == '__main__':
    # Example usage for testing
    test_query = "Plasmid construction using Golden Gate Assembly"
    test_block = """
------------------------------------------------------------------------
### [UHW250 Nucleic Acid Purification]
#### Meta
- Experimenter: AI Assistant
- Start_date: '2025-09-12 15:30'
- End_date: ''
#### Input
- (samples from the previous step) 
#### Reagent
- (e.g. enzyme, buffer, etc.) 
#### Consumables
- (e.g. filter, well-plate, etc.) 
#### Equipment
- (e.g. centrifuge, spectrophotometer, etc.) 
#### Method
- (method used in this step) 
#### Output
- (samples to the next step) 
#### Results & Discussions
- (Any results and discussions. Link file path if needed)
------------------------------------------------------------------------
"""
    
    print("--- Testing Method Agent ---")
    results = run_agent_team(test_query, test_block, "Method")
    print(results)