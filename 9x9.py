from openai import OpenAI
from tqdm import trange
import sudokum
import numpy as np
import re

def format_sudoku(puzzle):
    return str(np.array(puzzle))

def parse_sudoku(text):
    puzzle = [[]]
    for number in re.findall(r'\d', text):
        if len(puzzle[-1]) == 9:
            puzzle.append([])
        puzzle[-1].append(int(number))
    return np.array(puzzle)
        

client = OpenAI()
def run_llm(puzzle):
    prompt = "Solve this 9x9 sudoku by replacing 0s with numbers 1 to 9. Only print the solution, no other text." + "\n\n" + "The sudoku is:" + "\n\n" + format_sudoku(puzzle)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user",
                   "content": prompt 
                   }],
    )
    answer = completion.choices[0].message.content
    answer = parse_sudoku(answer)
    return answer


N = 10
cnt_correct = 0
cnt = 0
accuracy = 0
iterator = trange(N)
for line in iterator:
    puzzle = sudokum.generate(mask_rate=0.275)
    _, solution = sudokum.solve(puzzle)
    puzzle, solution = np.array(puzzle), np.array(solution)
    
    answer = run_llm(puzzle)
    correct = np.all(answer == solution)
    cnt_correct += correct
    cnt += 1
    accuracy = cnt_correct / cnt
    iterator.set_description(f"Accuracy: {accuracy:.3f} ({cnt_correct}/{cnt})")
