import os
import re
import logging
import ollama
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def _post_process_content(content: str) -> str:
    """
    LLM 응답에서 불필요한 접두사, 제목, 마크다운 블록을 제거하는 후처리 함수.
    """
    # "The answer is:", "The equipment is:" 등과 같은 불필요한 서두 제거
    content = re.sub(r'^(the\s*answer\s*is:|equipment:|method:)\s*', '', content, flags=re.IGNORECASE).strip()

    # 불필요한 마크다운 코드 블록 제거
    if content.startswith("```") and "```" in content[3:]:
        content = re.sub(r'^```[a-zA-Z]*\n', '', content)
        content = re.sub(r'\n```$', '', content)
        
    return content.strip()


async def call_llm_api(system_prompt: str, user_prompt: str, model_name: str = None):
    """LLM API를 호출하는 범용 비동기 함수."""
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", "biollama3")

    logger.info(f"Calling LLM: {model_name} for a specific task.")
    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        client = ollama.AsyncClient(host=ollama_base_url)

        response = await client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.1, 'top_p': 0.8}
        )
        content = response['message']['content'].strip()
        
        # 후처리 함수 호출
        processed_content = _post_process_content(content)
        return processed_content

    except Exception as e:
        logger.error(f"LLM API call failed: {e}", exc_info=True)
        return f"(LLM Error: Could not generate content due to: {e})"