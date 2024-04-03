import re
from typing import List, Tuple
import requests
from pathlib import Path
from ollama import Client

oclient = Client(host='http://localhost:11434')

def parse_qa(text: str) -> List[Tuple[str, str]]:
    """
    Parses text into a list of question-answer pairs, supporting multiline questions.
    
    Parameters:
        text (str): The input text containing questions and answers.
    
    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains a question and its corresponding answer.
    """
    # Initialize variables
    qa_pairs = []
    current_q = ""
    current_a = ""
    collecting_q = False  # Flag to indicate if we are currently collecting question lines

    # Process each line
    for line in text.split('\n'):
        if line.startswith('#'):
            if collecting_q:
                # Continue appending to current question
                current_q += line[1:].strip() + " "
            else:
                # New question encountered, save previous Q&A if exists
                if current_q and current_a:
                    qa_pairs.append((current_q.strip(), current_a.strip()))
                    current_a = ""  # Reset current answer
                current_q = line[1:].strip() + " "  # Start new question
                collecting_q = True
        else:
            # Not a question line, switch to collecting answer
            collecting_q = False
            current_a += line + "\n"
    
    # Don't forget to add the last Q&A pair if the text ends without a new question
    if current_q and current_a:
        qa_pairs.append((current_q.strip(), current_a.strip()))

    return qa_pairs


def get_text(textfile: str) -> str:
    if Path(textfile).exists():
        print("Found code snippets")
        with open(textfile, 'r') as fr:
            return fr.read()
    else:
        print("Downloading and saving code snippets.")
        code_data = 'https://github.com/akashe/Python-Code-Generation/raw/main/data/english_python_data_pruned.txt'
        content = requests.get(code_data).content
        with open(".python-code.txt", "wb") as fw:
            fw.write(content)
        return content.decode('utf-8')
 
def translate() -> None:
    textfile = ".python-code.txt"
    text = get_text(textfile) 
    print(f"Analyzing:")
    qa_pairs = parse_qa(text)
    fw = open('julia_code.txt', 'w')
    for q, a in qa_pairs:
        print(f"Question: {q}")
        message_template = f"""Translate the following snippit to the Julia programming language. Use only native libraries. Do not write anything except for the code. No explanation, or anything other than the code surrounded by ```  code  ``` blocks.
            ```
            {q}

            {a}
            ```
            """

        response = oclient.chat(model='llama2', messages=[
          {
            'role': 'user',
            'content': message_template
        },
        ])
        code = response['message']['content']
        print(code)
        regex = re.findall(r'```(.+?)```', code, re.DOTALL)
        if regex:
            snippet = f"""{q}
            {regex[-1]}
            """
            fw.write(snippet)
            fw.flush()
        else:
            raise RuntimeError("Could not parse code block.")
    fw.close()
