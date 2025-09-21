import os
import re
import logging
import asyncio
import json
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
    # ⭐️ 변경점: Supervisor Agent를 위한 상태 추가
    drafts: List[Dict[str, str]] # [{'model': 'biollama3', 'content': '...'}, ...]
    feedback: str # Supervisor의 재작성 요구사항
    final_options: List[str] # 최종 사용자에게 보여줄 옵션
    messages: Annotated[list, add_messages]


# --- Helper function for content extraction ---
def _extract_section_content(uo_block: str, section_name: str) -> str:
    """Helper to extract content of a specific section from a UO block."""
    pattern = re.compile(r"#### " + re.escape(section_name) + r"\n(.*?)(?=\n####|\n------------------------------------------------------------------------)", re.DOTALL)
    match = pattern.search(uo_block)
    if match:
        content = match.group(1).strip()
        return content if content and not content.startswith('(') else "(not specified)"
    return "(not specified)"

async def _generate_drafts(state: AgentState) -> AgentState:
    """
    Specialist Agent들의 역할을 수행하는 함수.
    여러 LLM을 동시에 호출하여 섹션에 대한 초안들을 생성합니다.
    """
    query = state['query']
    uo_id = state['uo_id']
    uo_name = state['uo_name']
    section = state['section_to_populate']
    uo_block = state['uo_block']
    feedback = state.get('feedback', '') # 재작성 시 피드백 활용

    logger.info(f"Generating drafts for UO '{uo_id}' - Section '{section}'")
    input_context = _extract_section_content(uo_block, "Input")
    rag_query = f"Find the specific procedure or list of items for the '{section}' section of the unit operation '{uo_id}: {uo_name}' related to the experiment: {query}"

    context_docs = rag_pipeline.retrieve_context(rag_query, k=3)
    rag_context = rag_pipeline.format_context_for_prompt(context_docs)

    base_user_prompt = f"""
- **Experiment Goal**: '{query}'
- **Unit Operation**: '{uo_id}: {uo_name}'
- **Section to Write**: '{section}'
- **Inputs**: '{input_context}'
"""
    if "No relevant context found" not in rag_context:
        base_user_prompt += f"\n--- **Relevant SOP Context** ---\n{rag_context}\n---"

    # 재작성 요청이 있을 경우 프롬프트에 피드백 추가
    if feedback:
        base_user_prompt += f"\n**IMPORTANT FEEDBACK FOR REVISION**: {feedback}\nPlease regenerate the content reflecting this feedback."


    system_prompt = "You are a specialized scientific assistant. Your task is to generate a comprehensive and well-structured response for a specific section of a lab note, using the provided context. The response should be clear, detailed, and directly applicable to the experiment. Your answer MUST be only the list or method itself, without any extra conversation or explanation."

    models_to_use = ["biollama3", "mixtral", "llama3:70b"]
    tasks = [
        call_llm_api(system_prompt, base_user_prompt, model_name)
        for model_name in models_to_use
    ]
    
    generated_contents = await asyncio.gather(*tasks)
    
    drafts = []
    for model_name, content in zip(models_to_use, generated_contents):
        if content and not content.startswith("(LLM Error"):
            drafts.append({'model': model_name, 'content': content})

    state['drafts'] = drafts
    return state


async def supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor Agent. 생성된 초안들을 평가하고 다음 단계를 결정합니다.
    """
    logger.info(f"Supervisor Agent: Evaluating drafts for UO '{state['uo_id']}' - Section '{state['section_to_populate']}'")
    drafts = state['drafts']
    if not drafts:
        logger.warning("Supervisor: No drafts to evaluate. Ending.")
        state['final_options'] = ["AI가 초안을 생성하지 못했습니다. 다시 시도해주세요."]
        return state

    # 평가를 위한 프롬프트 구성
    evaluation_prompt = """
You are a highly experienced principal investigator reviewing lab notes. Evaluate the following drafts for the '{section}' section of a protocol. For each draft, provide a score (out of 10) and a brief justification based on these criteria:
1.  **Structural Integrity (구조적 완성도)**: Is the format (e.g., Markdown list, numbered steps) clear and well-organized?
2.  **Specificity and Detail (내용의 구체성)**: Does it include specific, quantitative details like reagent concentrations, times, equipment models, etc.?
3.  **SOP Relevance (SOP 연관성)**: How well does it incorporate information from the provided SOP context?

**Format your response strictly as a JSON object, like this example:**
[
  {{"draft_index": 0, "model": "biollama3", "score": 8.5, "justification": "Clear steps, but lacks specific buffer concentrations."}},
  {{"draft_index": 1, "model": "mixtral", "score": 7.0, "justification": "Too generic and misses key details from the SOP."}},
  {{"draft_index": 2, "model": "llama3:70b", "score": 9.2, "justification": "Excellent detail and structure, accurately reflects the SOP."}}
]

--- DRAFTS TO EVALUATE ---
{draft_texts}
"""
    draft_texts = "\n\n---\n\n".join([f"**Draft {i} (from {d['model']})**:\n{d['content']}" for i, d in enumerate(drafts)])
    
    # llama3:70b를 채점자로 사용
    scoring_llm = "llama3:70b"
    logger.info(f"Calling Scoring LLM ({scoring_llm}) to evaluate drafts.")
    
    response_str = await call_llm_api(
        system_prompt="You are an expert lab note reviewer. Your output must be a valid JSON array of objects.",
        user_prompt=evaluation_prompt.format(section=state['section_to_populate'], draft_texts=draft_texts),
        model_name=scoring_llm
    )
    
    try:
        # LLM의 응답에서 JSON만 추출
        json_match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON array found in the LLM response.", response_str, 0)
        evaluations = json.loads(json_match.group(0))
        logger.info(f"Supervisor: Parsed evaluations: {evaluations}")
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Supervisor: Failed to parse JSON from scoring LLM. Error: {e}. Response: {response_str}")
        # 평가 실패 시, 원본 초안들을 그대로 사용
        state['final_options'] = [f"--- {d['model']}의 제안 ---\n\n{d['content']}" for d in drafts]
        state['feedback'] = '' # 피드백 없음
        return state

    # 점수가 가장 높은 초안 찾기
    best_draft_eval = max(evaluations, key=lambda x: x.get('score', 0))
    highest_score = best_draft_eval.get('score', 0)
    
    # 품질 기준(8.5점)을 통과했는지 확인
    if highest_score >= 8.5:
        logger.info(f"Supervisor: Quality threshold passed with score {highest_score}. Finalizing options.")
        # 고품질 초안들만 필터링하여 사용자에게 제공
        high_quality_drafts = [
            drafts[e['draft_index']] for e in evaluations if e.get('score', 0) >= 8.0
        ]
        state['final_options'] = [f"--- {d['model']}의 제안 (품질 점수: {next(e['score'] for e in evaluations if e['draft_index'] == i)}) ---\n\n{d['content']}" for i, d in enumerate(drafts) if d in high_quality_drafts]
        state['feedback'] = '' # 재작성 필요 없음
    else:
        logger.info(f"Supervisor: Quality threshold NOT passed (highest score: {highest_score}). Requesting revision.")
        # 재작성을 위한 피드백 생성
        feedback_points = [f"Draft from {e['model']} was critiqued: '{e['justification']}'" for e in evaluations]
        state['final_options'] = [] # 최종 옵션 없음
        state['feedback'] = f"The previous drafts were not detailed enough (top score was {highest_score}). Specific feedback: {' '.join(feedback_points)}. Please generate a much more detailed and specific version."

    return state


# --- Agent Nodes ---
async def specialist_agent_node(state: AgentState) -> AgentState:
    # 비동기 함수를 LangGraph 노드에서 실행하기 위해 await 사용
    return await _generate_drafts(state)

async def supervisor_agent_node(state: AgentState) -> AgentState:
    return await supervisor_agent(state)

# --- Routing Logic ---
def route_after_supervision(state: AgentState) -> str:
    if state.get('feedback'):
        logger.info("Routing: Feedback exists. Looping back to specialist agents.")
        return "specialist_agents"
    else:
        logger.info("Routing: No feedback. Proceeding to end.")
        return END

# --- Graph Definition ---
def create_agent_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("specialist_agents", specialist_agent_node)
    graph.add_node("supervisor", supervisor_agent_node)
    
    graph.set_entry_point("specialist_agents")
    graph.add_edge("specialist_agents", "supervisor")
    
    # Supervisor 평가 후 조건부 라우팅
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervision,
        {
            "specialist_agents": "specialist_agents",
            END: END
        }
    )
    
    agent_graph = graph.compile()
    logger.info("Supervisor-led agent graph compiled successfully.")
    return agent_graph

# --- Main execution function ---
def run_agent_team(query: str, uo_block: str, section: str) -> Dict:
    # 정규식에 \\? 를 추가하여 `[` 와 `\[` 를 모두 처리하도록 변경
    match = re.search(r"### \\?\[(U[A-Z]{2,3}\d{3,4}) (.*?)\\?\]", uo_block)
    if not match:
        logger.error(f"Could not parse UO ID and Name from block. UO Block Snippet:\n---\n{uo_block[:200]}\n---")
        # 오류 발생 시에도 Pydantic 모델이 요구하는 키를 포함하여 반환
        return {
            "uo_id": "Error",
            "section": section,
            "options": ["Error: Could not identify the Unit Operation. Please check the markdown format."]
        }
        
    uo_id, uo_name = match.groups()

    initial_state = AgentState(
        query=query,
        uo_block=uo_block,
        uo_id=uo_id,
        uo_name=uo_name,
        section_to_populate=section,
        drafts=[],
        feedback='',
        final_options=[],
        messages=[]
    )
    
    graph = create_agent_graph()
    # 비동기 그래프 실행
    final_state = asyncio.run(graph.ainvoke(initial_state))
    
    return {
        "uo_id": uo_id,
        "section": section,
        "options": final_state.get('final_options', [])
    }