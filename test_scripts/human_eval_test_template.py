import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import subprocess

def extract_code(gc):
  # extract code from the generated text
  # we can use regex to extract the code between ```python and ```
  import re

  # First, try to extract code between # BEGIN SOLUTION and # END SOLUTION
  solution_pattern = r"# BEGIN SOLUTION(.*?)# END SOLUTION"
  solution_match = re.search(solution_pattern, gc, re.DOTALL)
  if solution_match:
    return solution_match.group(1).strip()

  # Then try to extract code between ```python and ```
  pattern = r"```python(.*?)```"
  match = re.search(pattern, gc, re.DOTALL)
  if match:
    return match.group(1).strip()

  # Finally, look for code starting with import/def/from
  code_start = re.search(r'^(import|def|from)', gc, re.MULTILINE)
  if code_start:
      return gc[code_start.start():].strip()

  return gc.strip()

def extract_result(output):
  # extract the result from the output, we can look for the line that starts with "Passed" and extract the numbers
  import re
  pattern = r"Passed (\d+) out of (\d+) test cases"
  match = re.search(pattern, output)
  if match:
    pass_count = int(match.group(1))
    total_count = int(match.group(2))
    return pass_count, total_count
  else:
    return 0, 0

def check_main(gc):
  # check if the generated code has a main block that runs the tests
  return "if __name__ == \"__main__\""  and "def main():" in gc

def add_main_block(gc, test_cases, function_name):
  # the test_cases are like this, i need to replace change the candidate to the function name and then add a main block that runs the tests
  #   METADATA = {
  # 'author': 'jt',
  # 'dataset': 'test'
  # }


  # def check(candidate):
  # assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
  # assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
  # assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
  # assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
  # assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
  # assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
  # assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
  # we can replace "candidate" with the function name and then add a main block that runs the tests
  test_cases = test_cases.replace("candidate", function_name)
  # need to add the test_cases to the generated code and then add a main block that runs the tests with the proper intendation
  test_cases = "\n".join(["    " + line for line in test_cases.split("\n")])
  # can add a line to return how many of them passed from the test_cases
  # first line we can add a count thing
  test_cases = "pass_count = 0\n" + test_cases
  # then we can add a line to increment the count for each assert statement, we can do this by replacing "assert" with "if not (condition): pass else: pass_count += 1"
  test_cases = test_cases.replace("assert ", "if not (") + "): pass\nelse: pass_count += 1\n"
  test_cases = test_cases + "\ntotal_count = " + str(test_cases.count("assert "))
  # need to return the pass_count and total_count at the end of the test_cases
  test_cases = test_cases + "\nprint(f\"Passed {pass_count} out of {total_count} test cases\")"
  gc = gc + "\n\n" + test_cases
  main_block = f"""
  if __name__ == "__main__":
      try:
          check({function_name})
      except Exception as e:
    """
  return gc + "\n" + main_block

def evaulate_model_on_humaneval(model_name):
  # model id, for now: Qwen/Qwen3-4B
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

  dataset = load_dataset("openai/openai_humaneval", split="test")

  # dataset has these columns -> task_id, prompt, canonical_solution, test, entry_point we can ignore canonical_solution and entry_point for now
  # my idea is to generate the whole code from the prompt, save it and then run the test cases on it and see if it passes or not

  # prompt format
  # from typing import List


  # def has_close_elements(numbers: List[float], threshold: float) -> bool:
  # """ Check if in given list of numbers, are any two numbers closer to each other than
  # given threshold.
  # >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
  # False
  # >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
  # True
  # """

  # test format
  # METADATA = {
  # 'author': 'jt',
  # 'dataset': 'test'
  # }


  # def check(candidate):
  # assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
  # assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
  # assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
  # assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
  # assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
  # assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
  # assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

  prompt_template = """
  You are a Python expert. Your task is to complete the function below.
  Return ONLY the executable Python code.
  DO NOT include any explanation, markdown formatting outside of the code block, or introductory text.

  ### Instruction:
  Complete the function and include a main block that runs the provided test cases.
  The script must print the exact string: "Passed X out of Y test cases".

  ### Code to Complete:
  """

  prompt_end_thing = """
  ### Formatting Requirement:
  Include the following logic at the end of your script:
  if __name__ == "__main__":
      # run all test cases provided
      # ...
      print(f"Passed {pass_count} out of {total_count} test cases")
  """

  os.makedirs("temp_test", exist_ok=True)

  count_passed = 0

  for i in range(1):
    _, prompt, _, test_cases, _ = dataset[i].values()
    full_prompt = f"""# Task: Complete the Python function and run the tests.
      # Return ONLY the code.

      {prompt}
          # Implementation goes here
          pass

      {test_cases}

      if __name__ == "__main__":
          try:
              check({dataset[i]['entry_point']})
              # If no assertion error, all tests passed.
              # HumanEval 'test' strings usually contain multiple asserts.
              # We will wrap the 'check' function to count successes.
              print("Passed 1 out of 1 test cases")
          except Exception as e:
              print("Passed 0 out of 1 test cases")
      """
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=1024, temperature=0.1, top_p=0.95)
    print(f"outputs: {outputs}")
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated code for problem {i}:\n{generated_code}\n{'-'*50}")

    with open(f"temp_test/test_{i}.py", "w") as f:
      generated_code = extract_code(generated_code)
      x = check_main(generated_code)
      function_name = dataset[i]['entry_point']
      if not x:
        generated_code = add_main_block(generated_code, test_cases, function_name)
      f.write(generated_code)

    # need to run the generated code and check how many test cases pass, we can use subprocess to run the code and capture the output
    result = subprocess.run(["python", f"temp_test/test_{i}.py"], capture_output=True, text=True)
    print(result.stdout)

    res = extract_result(result.stdout)
    if res:
      pass_count, total_count = res
      if pass_count == total_count and total_count > 0:
        count_passed += 1

  print(f"Model {model_name} passed {count_passed} out of {len(dataset)} problems")

if __name__ == "__main__":
  model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
  evaulate_model_on_humaneval(model_name)