from openai import OpenAI
from tqdm import tqdm

path = '4x4-Sudoku-Dataset/4x4_sudoku_unique_puzzles.csv'

def format_sudoku(puzzle):
    s = ""
    for i in range(4):
        s += puzzle[i*4:(i*4)+4]
        s += "\n"
    return s

def parse_sudoku(text):
    return text.strip().replace("\n", "").strip()
        
def print_sudoku(puzzle):
    print(format_sudoku(puzzle))

with open(path) as f:
    lines = f.readlines()[1:]

client = OpenAI()
def run_llm(puzzle):
    prompt = "Solve this 4x4 sudoku by replacing 0s with numbers 1 to 4. Only print the solution, no other text." + "\n\n" + format_sudoku(puzzle) + "\n\n" + "The sudoku is:" + "\n\n" + format_sudoku(solution)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user",
                   "content": prompt 
                   }],
    )
    answer = completion.choices[0].message.content
    answer = parse_sudoku(answer)
    return answer


lines = lines[:10]
cnt_correct = 0
cnt = 0
accuracy = 0
iterator = tqdm(lines)
for line in iterator:
    puzzle, solution = line.split(',')
    puzzle, solution = puzzle.strip(), solution.strip()
    answer = run_llm(puzzle)
    correct = answer == solution
    cnt_correct += correct
    cnt += 1
    accuracy = cnt_correct / cnt
    iterator.set_description(f"Accuracy: {accuracy:.3f} ({cnt_correct}/{cnt})")
