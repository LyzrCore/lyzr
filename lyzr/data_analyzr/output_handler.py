import re
import logging
from typing import Literal
from itertools import chain

all_tasks = {
    "clean_data": [
        "remove_nulls",
        "convert_to_datetime",
        "convert_to_numeric",
        "convert_to_categorical",
    ],
    "transform": [
        "one_hot_encode",
        "ordinal_encode",
        "scale",
        "extract_time_period",
        "select_indices",
    ],
    "math_operation": ["add", "subtract", "multiply", "divide"],
    "analysis": [
        "sortvalues",
        "filter",
        "mean",
        "sum",
        "cumsum",
        "groupby",
        "correlation",
        "regression",
        "classification",
        "cluster",
        "forecast",
    ],
}

# ------------------------------ LLM output extraction --------------------------------------


def extract_sql(llm_response: str, logger: logging.Logger) -> str:
    # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
    sql = re.search(r"```sql\n(.*)```", llm_response, re.DOTALL)
    if sql:
        logger.info(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
        return sql.group(1)

    sql = re.search(r"```(.*)```", llm_response, re.DOTALL)
    if sql:
        logger.info(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
        return sql.group(1)

    return llm_response


# ------------------------------ LLM output validation --------------------------------------


def validate_output_step_details(step_details: dict, logger: logging.Logger):
    task_details, step_details = _check_task_details(step_details)
    if not task_details:
        logger.error("Invalid task details: {}.".format(step_details))
        raise ValueError("Invalid task details: {}".format(step_details))
    if not _check_task_and_type(step_details):
        logger.error(
            "Invalid task: {}. Invalid task details: {}.".format(
                step_details["task"], step_details
            )
        )
        raise ValueError(
            "Invalid task: {}. Invalid task details: {}.".format(
                step_details["task"], step_details
            )
        )
    return step_details


def _get_task_from_type(task: str, task_type: str) -> tuple:
    all_task_types = list(chain.from_iterable(all_tasks.values()))
    if (task_type not in all_task_types) and (task in all_task_types):
        task_type = task
    for task, types in all_tasks.items():
        if task_type in types:
            return task, task_type


def _check_task_details(step_details: dict) -> tuple:
    if (
        "task" in step_details
        and "type" in step_details
        and "args" in step_details
        and isinstance(step_details["args"], dict)
    ):
        return True, step_details
    if "task" not in step_details and "type" in step_details:
        step_details["task"] = ""
        return True, step_details
    if "type" not in step_details and "task" in step_details:
        if step_details["task"] in list(chain.from_iterable(all_tasks.values())):
            step_details["type"] = step_details["task"]
            return True, step_details
    if "args" not in step_details or not isinstance(step_details["args"], dict):
        step_details["args"] = {}
        return True, step_details
    return False, step_details


def _check_task_and_type(step_details: dict) -> bool:
    if (step_details["task"] not in all_tasks.keys()) or (
        step_details["type"] not in all_tasks[step_details["task"]]
    ):
        step_details["task"], step_details["type"] = _get_task_from_type(
            step_details["task"], step_details["type"]
        )
    if step_details["task"] not in all_tasks.keys():
        return False
    return True


# ---------------------------------- LLM output format checking ------------------------------------------------


# for output format dict
def check_output_format(
    llm_output: str,
    logger: logging.Logger,
    output_type: Literal["plot", "analysis"] = "analysis",
) -> dict:
    try:
        llm_output = eval(llm_output)
        logger.info("LLM output format checked.\n")
        return llm_output
    except NameError as e:
        e = str(e)
        if e.startswith("name") and e.endswith("is not defined"):
            word = re.findall(r"['\"](.*)['\"]", e)[0]
            llm_output = _check_word(llm_output, word)
            return check_output_format(llm_output, logger)
    except Exception as e:
        logger.info("Invalid output from LLM. Attempting to fix.")

    if output_type == "analysis":
        # remove all spaces in string
        llm_output = re.sub(r"\s+", "", llm_output)
        # string checking
        n_steps = _get_number_of_steps(llm_output)
        steps_components = _get_component_elements(llm_output)
        error_components = _check_num_components(steps_components, n_steps)
        if len(error_components) != 0:
            logger.info(f"Fixing components: {', '.join(error_components)}.")
            llm_output = _fix_components(error_components, steps_components)

        return check_output_format(llm_output, logger)


def _get_component_elements(llm_output):
    # string matching
    quotes = r"('|\")"
    number_regex = r"\d+"
    word_regex = rf"{quotes}\w+{quotes}"
    alphanum_regex = rf"({word_regex}|{number_regex})"
    list_0_regex = rf"\[{alphanum_regex}(,{alphanum_regex})*?\]"
    tasks_regex = r"['\"]task['\"]:(.*?),"
    task_types_regex = r"['\"]type['\"]:(.*?),"
    output_columns_regex = (r"['\"]outputcolumns['\"]:(.*)", r"['\"](\w+)['\"]")
    output_columns_list_regex = rf"{quotes}outputcolumns{quotes}:(\[{word_regex}(,{word_regex})+?\])"  # list of alphanumeric strings
    args_dict_regex = (
        "{"
        + f"{word_regex}:({word_regex}|{number_regex}|{list_0_regex})(,{word_regex}:({word_regex}|{number_regex}|{list_0_regex}))*?"
        + "}"  # dict with string keys and alphanum or list_0 values
    )
    step_details_dict_regex = (
        "{"
        + f"{word_regex}:({word_regex}|{number_regex}|{list_0_regex}|{args_dict_regex})(,{word_regex}:({word_regex}|{number_regex}|{list_0_regex}|{args_dict_regex}))*?"
        + "}"
    )  # dict with string keys and alphanum or list_0 or args_dict values
    all_steps_regex = rf"\[{step_details_dict_regex}(,{step_details_dict_regex})*?\]"  # list of step_details_dict
    response_dict_regex = (
        "{"
        + f"'steps':{all_steps_regex},'outputcolumns':{output_columns_list_regex}"
        + "}"
    )  # dict with keys 'steps' and 'output columns', and values all_steps and output_columns_list respectively

    steps_components = {
        "tasks": _get_tasks(llm_output, tasks_regex),
        "task_types": _get_task_types(llm_output, task_types_regex),
        "output_columns": _get_output_columns(llm_output, output_columns_regex),
        "output_columns_list": _get_output_columns_list(
            llm_output, output_columns_list_regex
        ),
        "args_dict": _get_args_dicts(llm_output, args_dict_regex),
        "step_details_dict": _get_steps_dicts(llm_output, step_details_dict_regex),
        "all_steps": _get_all_steps(llm_output, all_steps_regex),
        "response_dict": _get_response_dict(llm_output, response_dict_regex),
    }
    return steps_components


def _get_number_of_steps(llm_output):
    number = r"(\w+)"
    num_steps = []
    for word in re.finditer(number, llm_output):
        try:
            num_steps.append(int(word.group()))
        except ValueError:
            llm_output = _check_word(llm_output, word)
    while len(num_steps) != 0:
        if len(num_steps) != max(num_steps):
            # remove highest number and try again
            num_steps.remove(max(num_steps))
        else:
            return max(num_steps)
    return 0


def _check_word(llm_output, word):
    if isinstance(word, str):
        nmatches = len([x for x in re.finditer(word, llm_output)])
        for idx in range(nmatches):
            match = [x for x in re.finditer(word, llm_output)][idx]
            llm_output = _check_word_match(llm_output, match)
        return llm_output
    elif isinstance(word, re.Match):
        return _check_word_match(llm_output, word)


def _check_word_match(llm_output, word):
    start_quote = llm_output[word.start() - 1]
    end_quote = llm_output[word.end()]
    if start_quote not in ["'", '"']:
        if end_quote not in ["'", '"']:
            llm_output = llm_output[: word.start()] + "'" + llm_output[word.start() :]
            llm_output = (
                llm_output[: word.end() + 1] + "'" + llm_output[word.end() + 1 :]
            )
        else:
            llm_output = (
                llm_output[: word.start()] + end_quote + llm_output[word.start() :]
            )
    elif end_quote not in ["'", '"']:
        llm_output = llm_output[: word.end()] + start_quote + llm_output[word.end() :]
    return llm_output


def _check_num_components(steps_components, n_steps):
    error_components = []
    for component in steps_components:
        if component == "output_columns":
            continue
        if component in ["tasks", "task_types", "args_dict", "step_details_dict"]:
            expected_num_components = n_steps
        else:
            expected_num_components = 1
        if len(steps_components[component]) != expected_num_components:
            error_components.append(component)
    return error_components


def _fix_components(error_components, steps_components):
    for component in error_components:
        if component in ["tasks", "task_types", "args_dict", "output_columns"]:
            raise ValueError("Cannot fix LLM output.")
        steps_components[component] = globals()[f"_fix_{component}"](steps_components)
    return steps_components["response_dict"][0]


def _fix_step_details_dict(steps_components):
    step_details = []
    for step, args_dict in enumerate(steps_components["args_dict"]):
        step_details.append(
            "{"
            + f"'step':{step},'task':{steps_components['tasks'][step]},'type':{steps_components['type'][step]},'args':{args_dict}"
            + "}"
        )


def _fix_output_columns_list(steps_components):
    return ["[" + ",".join(steps_components["output_columns"]) + "]"]


def _fix_all_steps(steps_components):
    return ["[" + ",".join(steps_components["step_details_dict"]) + "]"]


def _fix_response_dict(steps_components):
    return [
        "{"
        + f"'steps':{steps_components['all_steps'][0]},'output columns':{steps_components['output_columns_list'][0]}"
        + "}"
    ]


def _get_tasks(llm_output, tasks_regex):
    tasks = []
    for x in re.finditer(tasks_regex, llm_output):
        tasks.append(x.group(1))
    return tasks


def _get_task_types(llm_output, task_types_regex):
    task_types = []
    for x in re.finditer(task_types_regex, llm_output):
        task_types.append(x.group(1))
    return task_types


def _get_output_columns(llm_output, output_columns_regex):
    output_columns = []
    for x in re.finditer(output_columns_regex[0], llm_output):
        for y in re.finditer(output_columns_regex[1], x.group(1)):
            output_columns.append(y.group(1))
    return output_columns


def _get_output_columns_list(llm_output, output_columns_list_regex):
    output_columns_list = []
    for x in re.finditer(output_columns_list_regex, llm_output):
        output_columns_list.append(x.group(3))
    return output_columns_list


def _get_args_dicts(llm_output, args_dict_regex):
    args_dicts = []
    for x in re.finditer(args_dict_regex, llm_output):
        args_dicts.append(x.group())
    return args_dicts


def _get_steps_dicts(llm_outptut, steps_dicts_regex):
    steps_dicts = []
    for x in re.finditer(steps_dicts_regex, llm_outptut):
        steps_dicts.append(x.group())
    return steps_dicts


def _get_all_steps(llm_output, all_steps_regex):
    all_steps = []
    for x in re.finditer(all_steps_regex, llm_output):
        all_steps.append(x.group())
    return all_steps


def _get_response_dict(llm_output, response_dict_regex):
    response_dict = []
    for x in re.finditer(response_dict_regex, llm_output):
        response_dict.append(x.group())
    return response_dict
