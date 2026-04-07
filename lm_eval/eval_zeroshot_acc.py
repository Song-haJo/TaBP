import argparse
import json
import logging
import os

from . import utils_
from . import evaluator, tasks
from ._utils import convert_json2csv_zeroshot


logging.getLogger("openai").setLevel(logging.WARNING)

def eval_acc(
                    model,
                    tasks_list=None,
                    num_fewshot=0,
                    batch_size=None,
                    max_batch_size=None,
                    device=None,
                    output_json=None,
                    limit_percentage=None,
                    limit_fixed=None,
                    no_cache=False,
                    decontamination_ngrams_path=None,
                    description_dict_path=None,
                    check_integrity=False,
                    write_out=False,
                    output_base_path=None
                   ):
    

    if tasks_list is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils_.pattern_match(tasks_list.split(","), tasks.ALL_TASKS)
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if description_dict_path:
        with open(description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        # device=device,
        no_cache=no_cache,
        limit_percentage=limit_percentage,
        limit_fixed=limit_fixed,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        check_integrity=check_integrity,
        write_out=write_out,
        output_base_path=output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"limit_percentage: {limit_percentage}, limit_fixed: {limit_fixed}, num_fewshot: {num_fewshot}, batch_size: {batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))

    csv_path = output_json.replace(".json", ".csv") if output_json else None
    if csv_path:
        convert_json2csv_zeroshot(output_json, csv_path, task_names)
