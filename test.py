from vllm import LLM, SamplingParams


model_name=""
prompt=""

def load_model(model_name: str):
    llm = LLM(model=model_name, gpu_memory_utilization=0.90)
    return llm


teacher_model1=load_model(model_name)
teacher_model2=load_model(model_name)
teacher_model3=load_model(model_name)


def get_30_tokens(llm1: LLM, llm2: LLM, llm3: LLM,prompt: str):
    vocab_size = llm1.get_tokenizer().vocab_size
    params = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size
    )

    output1 = llm1.generate(prompts=[prompt], sampling_params=params)
    output2 = llm2.generate(prompts=[prompt], sampling_params=params)
    output3 = llm3.generate(prompts=[prompt], sampling_params=params)
    #First Prompt -> First Generated Sequence -> First Generated Token
    token_logprobs_dict1 = output1[0].outputs[0].logprobs[0]
    token_logprobs_dict2 = output2[0].outputs[0].logprobs[0]
    token_logprobs_dict3 = output3[0].outputs[0].logprobs[0]

    #Convert the dictionary to a list and sort it by logprob value
    sorted_logprobs1 = sorted(
        [(token_id, lp_obj.logprob) for token_id, lp_obj in token_logprobs_dict1.items()],
        key=lambda x: x[1]
    )

    sorted_logprobs2 = sorted(
        [(token_id, lp_obj.logprob) for token_id, lp_obj in token_logprobs_dict2.items()],
        key=lambda x: x[1]
    )

    sorted_logprobs3 = sorted(
        [(token_id, lp_obj.logprob) for token_id, lp_obj in token_logprobs_dict3.items()],
        key=lambda x: x[1]
    )

    bottom5_logprobs1 = sorted_logprobs1[:5]
    top5_logprobs1 = sorted_logprobs1[-5:][::-1]

    bottom5_logprobs2 = sorted_logprobs2[:5]
    top5_logprobs2 = sorted_logprobs2[-5:][::-1]

    bottom5_logprobs3 = sorted_logprobs3[:5]
    top5_logprobs3 = sorted_logprobs3[-5:][::-1]

    return top5_logprobs1+bottom5_logprobs1+top5_logprobs2+bottom5_logprobs2+top5_logprobs3+bottom5_logprobs3