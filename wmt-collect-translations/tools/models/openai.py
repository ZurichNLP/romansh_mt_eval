from dotenv import load_dotenv
import os

load_dotenv()

CLIENT = None
def lazy_get_client():
    global CLIENT

    if CLIENT is None:
        import openai
        CLIENT = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return CLIENT


def openai_gpt4o(prompt):
    return openai_call(prompt, "gpt-4o")

def openai_call(prompt, model, **kwargs):
    client = lazy_get_client()
    import openai

    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
            **kwargs,
        )
    except (openai.BadRequestError, openai.APITimeoutError) as e:
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
