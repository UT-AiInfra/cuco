import backoff
import anthropic
from .pricing import CLAUDE_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


MAX_TRIES = 20
MAX_VALUE = 20

# Bedrock (and long requests) require streaming to avoid "Streaming is required for
# operations that may take longer than 10 minutes" (Anthropic SDK).
def _use_streaming(client) -> bool:
    return getattr(client, "__class__", None) and (
        client.__class__.__name__ == "AnthropicBedrock"
    )


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    on_backoff=backoff_handler,
)
def query_anthropic(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Anthropic/Bedrock model."""
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for Anthropic.")

    use_streaming = _use_streaming(client)
    if use_streaming:
        # Bedrock/long requests: use stream to avoid 10-minute timeout.
        with client.messages.stream(
            model=model,
            system=system_msg,
            messages=new_msg_history,
            **kwargs,
        ) as stream:
            response = stream.get_final_message()
    else:
        response = client.messages.create(
            model=model,
            system=system_msg,
            messages=new_msg_history,
            **kwargs,
        )

    # Separate thinking from non-thinking content
    if len(response.content) == 1:
        thought = ""
        content = response.content[0].text
    else:
        thought = getattr(response.content[0], "thinking", "") or ""
        content = response.content[1].text

    new_msg_history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    )
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    input_cost = CLAUDE_MODELS.get(model, {}).get("input_price", 0) * input_tokens
    output_cost = CLAUDE_MODELS.get(model, {}).get("output_price", 0) * output_tokens
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
