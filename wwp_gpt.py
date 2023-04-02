import openai


def issue_command(command, text):
    api_result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": command},
            {"role": "user", "content": text},
        ]
    )

    if "choices" in api_result:
        # Take the first choice as the result
        content = api_result["choices"][0]["message"]["content"]
        print("Content: ", content)
        return content

    raise ValueError("No choices in result")
