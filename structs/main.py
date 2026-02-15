from sre_parse import Tokenizer
from vllm import LLM, SamplingParams
import math


model_name=""
c=1.414
def load_model(model_name: str):
    llm = LLM(model=model_name, gpu_memory_utilization=0.90)
    return llm


teacher_model1=load_model(model_name)
teacher_model2=load_model(model_name)
teacher_model3=load_model(model_name)
tokenizer1 = teacher_model1.get_tokenizer()
tokenizer2 = teacher_model2.get_tokenizer()
tokenizer3 = teacher_model3.get_tokenizer()

class Node:
  def __init__(self,token_id,generated_text,parent):
    self.token_id=token_id
    self.generated_text=generated_text
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
  prompt=f"For this prompt {prompt} you have generated these texts untill this {leaf_node.generated_text} now generate the next token with this context"
  generate_tokens=get_30_tokens(teacher_model1,teacher_model2,teacher_model3,prompt,tokenizer1,tokenizer2,tokenizer3)
  for token in generate_tokens:
    child_node=Node(token[0],leaf_node.generated_text+token[1],leaf_node)
    leaf_node.add_child(child_node)


def backpropagte(node , reward):
  node.visit_count+=1
  node.value+=reward
  if node.parent is not None :
    return backpropagte(node.parent , reward)




def get_30_tokens(llm1: LLM, llm2: LLM, llm3: LLM,prompt: str , tokenizer1 : Tokenizer, tokenizer2 : Tokenizer, tokenizer3 : Tokenizer):
    vocab_size1 = tokenizer1.vocab_size
    vocab_size2 = tokenizer2.vocab_size
    vocab_size3 = tokenizer3.vocab_size
    params1 = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size1
    )
    params2 = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size2
    )
    params3 = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size3
    )

    output1 = llm1.generate(prompts=[prompt], sampling_params=params1)
    output2 = llm2.generate(prompts=[prompt], sampling_params=params2)
    output3 = llm3.generate(prompts=[prompt], sampling_params=params3)
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
    def format_decoded_list(logprob_list, tokenizer):
      # Wraps token_id in a list [] because decode() expects a sequence of IDs
      return [(t_id, tokenizer.decode([t_id]), lp) for t_id, lp in logprob_list]

    bottom5_1 = format_decoded_list(sorted_logprobs1[:5], tokenizer1)
    top5_1 = format_decoded_list(sorted_logprobs1[-5:][::-1], tokenizer1)

    bottom5_2 = format_decoded_list(sorted_logprobs2[:5], tokenizer2)
    top5_2 = format_decoded_list(sorted_logprobs2[-5:][::-1], tokenizer2)

    bottom5_3 = format_decoded_list(sorted_logprobs3[:5], tokenizer3)
    top5_3 = format_decoded_list(sorted_logprobs3[-5:][::-1], tokenizer3)

    return top5_1 + bottom5_1 + top5_2 + bottom5_2 + top5_3 + bottom5_3



def mcts(prompt , num_simulations) :
  root_node=Node(token_id=None,generated_text="",parent=None)

  for _ in range(num_simulations) :
    leaf=select_leaf_node(root_node)
    expand_leaf(leaf,prompt)
    reward=0.5 # dummy
    backpropagte(leaf,reward)

  return root_node

