from openai import OpenAI
from tqdm import trange
import sudokum
import numpy as np
import re
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def llm(prompt, verbose=False):
    if verbose: print(f"> {prompt}")
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user",
                   "content": prompt 
                   }],
    )
    answer = completion.choices[0].message.content
    if verbose: print(f"< {answer}")
    return answer


def format_sudoku(puzzle):
    return str(np.array(puzzle))

def parse_sudoku(text):
    print(text)
    numbers = re.findall(r'\d', text)
    if len(numbers) != 81:
        raise ValueError("The parsed text does not contain exactly 81 digits.")
    puzzle = np.array(numbers, dtype=int).reshape((9, 9))
    return puzzle
        

client = OpenAI()
def run_llm(puzzle, prompt, few_shot=0):
    for _ in range(few_shot):
        p = sudokum.generate(mask_rate=0.1)
        _, s = sudokum.solve(p)
        p, s= np.array(p), np.array(s)
        prompt += "Here is an example:\n"
        prompt += format_sudoku(p)
        prompt += "The solution is:\n"
        prompt += format_sudoku(s)
        prompt += "\n\n"

    prompt += "The sudoku is:" + "\n\n" + format_sudoku(puzzle)
    answer = llm(prompt)
    answer = parse_sudoku(answer)
    return answer


N = 25
cnt_correct = 0
cnt = 0
accuracy = 0
iterator = trange(N)

prompt = """Solve this 9x9 sudoku by replacing 0s with numbers 1 to 9.
            Only print the solution, no other text.\n\n"""

# metaprompt = f"Refine the following instructions for a LLM solving a sudoku puzzle. Make sure to make it better at solving the task. ```{prompt}```"
metaprompt = f"""Enhance the given problem-solving instructions for an LLM:
Identify the problem type and its key characteristics.
Analyze the current prompt's strengths and weaknesses.
Incorporate relevant domain knowledge and best practices.
Develop a structured, step-by-step problem-solving strategy.
Encourage reflection on the problem-solving process for future improvement.
Refine the language for clarity and better LLM comprehension.
Current prompt:
```{prompt}```
Based on these points, provide an improved version of the prompt with a comprehensive problem-solving plan that enhances the LLM's ability to solve this type of problem effectively and accurately."""
prompt = llm(metaprompt, verbose=True)

for line in iterator:
    puzzle = sudokum.generate(mask_rate=0.3)
    _, solution = sudokum.solve(puzzle)
    puzzle, solution = np.array(puzzle), np.array(solution)
    
    answer = run_llm(puzzle, prompt=prompt, few_shot=0)
    correct = np.all(answer == solution)
    cnt_correct += correct
    cnt += 1
    accuracy = cnt_correct / cnt
    iterator.set_description(f"Accuracy: {accuracy:.3f} ({cnt_correct}/{cnt})")