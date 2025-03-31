import openai
MAX_RETRY = 5 # in case when you prompt an llm the api returns an error (it may sometime happen if you do multiple LLM calls in parallel)

def llama(
        user_prompt=None, 
        system_prompt=None, 
        n_used=1,
        logprobs=False,
        seed=None,
        temperature=0.7,
        top_p=0.95,
        llm_name='llama-3-70B'
    ):
    success = False
    it = 0

    if llm_name == 'llama-3-70B':
        llm_name_used = 'meta-llama/Meta-Llama-3-70B-Instruct'
    elif llm_name == 'llama-3-8B':
        llm_name_used = 'meta-llama/Meta-Llama-3-8B-Instruct'

    while not success and it < MAX_RETRY:
        it += 1
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        if system_prompt is None:
            system_prompt = "You are an AI assistant that helps people find information."
        message_text = [{"role":"system","content": system_prompt}]
        if user_prompt:
            message_text.append({"role":"user", "content": user_prompt})
        print("---------- Waiting-------------")
        response = client.chat.completions.create(
                model=llm_name_used,
                messages=message_text,
                n=n_used,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                seed=seed,
            )
        try:
            response.choices[0].message.content is not None
            success = True
            print("---------- PASS -------------")
        except:
            print("---------- NOT PASS -------------")
            pass

    return response