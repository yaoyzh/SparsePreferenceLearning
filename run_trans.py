import os
import click
from pathlib import Path
import subprocess
import inspect
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).parent


def run_shell_command(command, output_file="output.txt", to_terminal=True):

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  
            universal_newlines=True,
        )

        stdout_lines = []
        stderr_lines = []

        with open(output_file, "w", encoding="utf-8") as f:
            if to_terminal:
                print("=" * 80)
            for i, line in enumerate(process.stdout):
                formatted_line = f"{i}: {line.strip()}\n"
                if to_terminal:
                    print(formatted_line, end="")  
                f.write(formatted_line) 
                stdout_lines.append(line)

            for line in process.stderr:
                formatted_error = f"Error: {line.strip()}\n"
                if to_terminal:
                    print(formatted_error, end="")  
                f.write(formatted_error) 
                stderr_lines.append(line)

        return_code = process.wait()

        if return_code == 0:
            success_message = f"\nCommand({command}) has been executed successfully.\n Outputs have been saved in {output_file}."
            if to_terminal:
                print(success_message)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(success_message + "\n")
            return "\n".join(stdout_lines)
        else:
            failure_message = (
                f"\nCommand({command}) failed.\nError information has been saved in {output_file}."
            )
            if to_terminal:
                print(failure_message)
                print("Error:\n", "".join(stderr_lines))
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(failure_message + "\n")
            exit(return_code)

    except Exception as e:
        error_message = f"Abnormal: {e}"
        if to_terminal:
            print(error_message)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(error_message + "\n")
        exit(1)


@click.group()
def cli():
    pass

@cli.command()
def rmstaticllama1b():
    fn = inspect.currentframe().f_code.co_name
    model_name = "meta-llama/Llama-3.2-1B"
    dataset_name = "Dahoas/rm-static"
    entry_script = "trans.py" 

    output_dir = (
        SCRIPT_ROOT / "datahidden" / fn
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = f"""deepspeed --num_gpus 1 {entry_script} \
        --data_path {dataset_name} \
        --model_name_or_path {model_name} \
        --deepspeed \
        --output_dir {output_dir}"""

    run_shell_command(command, output_dir / "output.txt")



@cli.command()
def shpllama1b():
    fn = inspect.currentframe().f_code.co_name
    model_name = "meta-llama/Llama-3.2-1B"
    dataset_name = "stanfordnlp/SHP"
    entry_script = "trans.py" 

    output_dir = (
        SCRIPT_ROOT / "datahidden" / fn
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = f"""deepspeed --num_gpus 1 {entry_script} \
        --data_path {dataset_name} \
        --model_name_or_path {model_name} \
        --deepspeed \
        --output_dir {output_dir} \
        --data_split 0,37,963"""

    run_shell_command(command, output_dir / "output.txt")


if __name__ == "__main__":
    cli()
