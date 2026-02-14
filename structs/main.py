from vllm import LLM, SamplingParams
import math


model_name=""

def load_model(model_name: str):
    llm = LLM(model=model_name, gpu_memory_utilization=0.90)
    return llm


teacher_model1=load_model(model_name)
teacher_model2=load_model(model_name)
teacher_model3=load_model(model_name)


class Node:
  def __init__(self,token_id,pre_tokens,parent):
    self.token_id=token_id
    self.pre_tokens=pre_tokens
    self.parent=parent
    self.children=[]
    self.visit_count=0
    self.value=0.0

  def add_child(self,child_node):
    self.children.append(child_node)


  def get_ucb_score(self):
    """We cant use the puct formula here becuase we use child.prior there meaning we get log_probs tells you how confident it is in that token and based on this the exploration happens but here we need winning and losing traces so if the tokens are bad the logs are bad and mcts wont explore it that much..so use ucb1"""
    if self.visit_count>0:
      exploitation= self.value/self.visit_count
      exploration=c*(math.sqrt(math.log(self.parent.visit_count)/self.visit_count))
    else :
      return float('inf')
    return exploitation + exploration


def select_leaf_node(root_node):
  current_node=root_node
  while(len(current_node.children)!=0):
    next_node=None
    max_ucb=- float('inf')
    for i in range(0,len(current_node.children)):
      ucb_score=current_node.children[i].get_ucb_score()
      if ucb_score>max_ucb:
        max_ucb=ucb_score
        next_node=current_node.children[i]
    current_node=next_node
  return current_node



def expand_leaf(leaf_node,prompt: str):
  prompt="For this prompt {prompt} you have generated tokens untill this {leaf_node.pre_tokens} now generate the next token with this context"
  generate_token






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





# def check_full_generations():
#   return 0

# def tree():
#   model_name=""
#   teacher_model1=load_model(model_name)
#   teacher_model2=load_model(model_name)
#   teacher_model3=load_model(model_name)

#   for data in loaded_datasets :
#     prompt=data["prompt"]
#     top5_logprobs1, bottom5_logprobs1 = get_logprobs(teacher_model1, prompt)
#     top5_logprobs2, bottom5_logprobs2 = get_logprobs(teacher_model2, prompt)
#     top5_logprobs3, bottom5_logprobs3 = get_logprobs(teacher_model3, prompt)

#     if check_full_generations():



