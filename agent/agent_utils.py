"""
Utility functions for the agent.
"""

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately


def trim_history(messages, max_tokens: int = 5000):
    """
    Trim a list of chat messages to keep only the last messages up to ~max_tokens tokens.
    Uses `trim_messages` utility from LangChain.
    
    Args:
        messages: List of messages to trim
        max_tokens: Maximum number of tokens to keep (default: 5000)
        
    Returns:
        Trimmed list of messages
    """
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )


def reduce_messages(messages, keep_last_user=1, keep_last_ai=1):
    """
    Safe reducer that keeps the last user + AI messages
    AND preserves valid tool_call → tool_response ordering.
    
    Args:
        messages: List of messages to reduce
        keep_last_user: Number of last user messages to keep (default: 1)
        keep_last_ai: Number of last AI messages to keep (default: 1)
        
    Returns:
        Reduced list of messages with preserved tool call ordering
    """
    # Reverse iterate
    reversed_msgs = list(reversed(messages))

    kept = []
    user_count = 0
    ai_count = 0

    # Collect minimal relevant messages
    for msg in reversed_msgs:
        if isinstance(msg, HumanMessage) and user_count < keep_last_user:
            kept.append(msg)
            user_count += 1
        elif isinstance(msg, AIMessage) and ai_count < keep_last_ai:
            kept.append(msg)
            ai_count += 1

    # Determine which tool_call_ids must be preserved
    required_tool_ids = set()
    for msg in kept:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                required_tool_ids.add(tc["id"])

    # Collect matching ToolMessages
    tool_msgs = [
        msg for msg in messages
        if isinstance(msg, ToolMessage) and msg.tool_call_id in required_tool_ids
    ]

    # Build new message list in chronological order
    new_messages = []

    for msg in messages:
        # Add AI
        if msg in kept:
            new_messages.append(msg)

            # Immediately attach its tool responses
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    for tm in tool_msgs:
                        if tm.tool_call_id == tc["id"]:
                            new_messages.append(tm)

        # Add User only if in kept
        elif isinstance(msg, HumanMessage) and msg in kept:
            new_messages.append(msg)

    return new_messages

