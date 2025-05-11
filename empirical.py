import os
import click
from pathlib import Path
import subprocess
import inspect
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).parent


def check_experiment_finish_by_counting_acc(dir, ep=1):
    if not (dir / "output.txt").exists():
        return False
    if not (dir / "pytorch_model.bin").exists():
        return False
    with open(dir / "output.txt", "r", encoding="latin-1") as f:
        lines = f.readlines()
        counter = 0
        counter_sparse = 0
        for line in lines:
            if "acc (higher is better) :" in line:
                counter += 1
            if "self sparsity:" in line:
                counter_sparse += 1
    return counter == (2+ep-1) and counter_sparse == 1

def check_evaluation_finish_by_counting_acc(dir):
    if not (dir / "output_eval.txt").exists():
        return False
    with open(dir / "output_eval.txt", "r", encoding="latin-1") as f:
        lines = f.readlines()
        counter = 0
        counter_sparse = 0
        for line in lines:
            if "acc (higher is better) :" in line:
                counter += 1
    return counter == 1

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
            success_message = f"\nCommand({command}) has been successfully executed.\nResults have been saved to {output_file}."
            if to_terminal:
                print(success_message)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(success_message + "\n")
            return "\n".join(stdout_lines)
        else:
            failure_message = (
                f"\nCommand({command}) failed to execute.\nError messeges have been saved to {output_file}."
            )
            if to_terminal:
                print(failure_message)
                print("Error messege:\n", "".join(stderr_lines))
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(failure_message + "\n")
            exit(return_code)

    except Exception as e:
        error_message = f"Exception: {e}"
        if to_terminal:
            print(error_message)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(error_message + "\n")
        exit(1)


@click.group()
def cli():
    pass


@cli.command()
def rmstaticpythia70mlastsplr1em5wd1em1ep1():
    fn = inspect.currentframe().f_code.co_name
    model_name = "EleutherAI/pythia-70m"
    dataset_name = "Dahoas/rm-static"
    entry_script = "main.py" 
    wd = 1e-1
    lr = 1e-5
    ep = 1
    trials = 5
    l1_list = [0, 10**(-4.5), 10**(-4), 10**(-3.75), 10**(-3.5), 10**(-3.25), 10**(-3), 10**(-2.75), 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-0.5), 1, 2, 4, 8]
    for seed in [i for i in range(trials)]: 
        for l1_lambda in l1_list:
            output_dir = (
                SCRIPT_ROOT / "output" / fn / f"l1reg{l1_lambda}" / f"seed{seed}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            if check_experiment_finish_by_counting_acc(output_dir, ep=ep):
                print(f"Experiment already finished: {output_dir}")
                continue
            command = f"""deepspeed --num_gpus 1 {entry_script} \
                --data_path {dataset_name} \
                --model_name_or_path {model_name} \
                --deepspeed \
                --eval_interval 500 \
                --seed {seed} \
                --output_dir {output_dir} \
                --l1_lambda {l1_lambda} \
                --weight_decay {wd} \
                --learning_rate {lr} \
                --num_train_epochs {ep} \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8"""

            run_shell_command(command, output_dir / "output.txt")




if __name__ == "__main__":
    cli()
