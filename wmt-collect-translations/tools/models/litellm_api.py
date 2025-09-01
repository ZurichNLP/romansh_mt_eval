from dotenv import load_dotenv

load_dotenv()


def litellm_gemini_2_5_flash(prompt):
    return litellm_call(prompt, "litellm_proxy/gemini-2.5-flash", thinking={"type": "disabled", "budget_tokens": 0})

def litellm_llama_70b(prompt):
    return litellm_call(prompt, "litellm_proxy/vertex_ai/meta/llama-3.3-70b-instruct-maas")

def litellm_call(prompt, model, **kwargs):
    import litellm
    import openai
    # litellm._turn_on_debug()

    try:
        response = litellm.completion(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=8192,
            **kwargs,
        )
    except (openai.BadRequestError, openai.APITimeoutError) as e:
        print(e)
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise e


    if response.choices[0].finish_reason != "stop":
        return None

    assert response.choices[0].finish_reason == "stop", f"Finish reason: {response.choices[0].finish_reason}"

    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)
