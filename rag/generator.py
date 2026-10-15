from __future__ import annotations

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from config import NVIDIA_API_KEY, NVIDIA_LLM_MODEL

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    financial_summary: dict,
    ticker: str = '',
) -> str:
    """Generate an answer using NVIDIA LLM with retrieved context."""
    if not NVIDIA_API_KEY:
        return _fallback_answer(question, retrieved_chunks, financial_summary)
    
    llm = ChatNVIDIA(
        model=NVIDIA_LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=0.2,
        max_completion_tokens=1024,
    )
    
    # Format context
    context_text = "\n\n".join(
        f"Source: {chunk.get('metadata', {}).get('section_name', 'Unknown Section')} "
        f"(Year: {chunk.get('metadata', {}).get('year', 'Unknown Year')})\n"
        f"Text: {chunk.get('text', '')}"
        for chunk in retrieved_chunks
    )
    
    # Format financial summary
    summary_text = (
        f"Latest Year: {financial_summary.get('latest_year', 'N/A')}\n"
        f"Revenue: ${financial_summary.get('revenue', 0):,.0f}\n"
        f"Net Margin: {financial_summary.get('net_margin', 0) * 100:.1f}%\n"
        f"Current Ratio: {financial_summary.get('current_ratio', 0):.2f}\n"
        f"Debt to Equity: {financial_summary.get('debt_to_equity', 0):.2f}\n"
        f"Health Signal: {financial_summary.get('health_signal', 'N/A')}\n"
    )
    
    system_prompt = (
        "You are a financial analyst assistant. Answer the user's question based on the provided "
        "SEC filing excerpts and financial data.\n"
        "Please synthesize an answer and include citations from the context."
    )
    
    user_prompt = (
        f"Question: {question}\n\n"
        f"Ticker: {ticker}\n\n"
        f"Financial Summary:\n{summary_text}\n\n"
        f"Retrieved Excerpts:\n{context_text}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}\n\n" + _fallback_answer(question, retrieved_chunks, financial_summary)

def _fallback_answer(question: str, chunks: list[dict], summary: dict) -> str:
    """Simple non-LLM answer when NVIDIA key is not available."""
    text = f"Fallback Answer for {question}\n\n"
    text += "Financial Summary:\n"
    text += f"Revenue: ${summary.get('revenue', 0):,.0f}, Net Margin: {summary.get('net_margin', 0) * 100:.1f}%\n\n"
    
    if chunks:
        text += "Most relevant excerpt:\n"
        text += chunks[0].get('text', '')
    else:
        text += "No excerpts found."
        
    return text
