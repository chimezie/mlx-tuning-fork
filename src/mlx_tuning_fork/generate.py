import click
import json
from mlx_tuning_fork.config import CONFIG_DEFAULTS, get_prompt_formatter, PROMPT_FORMATS
from ogbujipt import word_loom
from ogbujipt.prompting import format
import mlx.core as mx
from mlx_lm.utils import load, generate

DEFAULT_SEED = 0


def colorprint(color, s):
    color_codes = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(s, t0):
    if t0 > 0.8:
        color = "white"
    elif t0 > 0.6:
        color = "cyan"
    elif t0 > 0.4:
        color = "green"
    elif t0 > 0.2:
        color = "yellow"
    else:
        color = "red"
    colorprint(color, s)


def generate_prompt_from_loom(loom_file, loom_markers, prompt_formatter, build_prompt, cot_source_path, tokenizer):
    with open(loom_file, mode='rb') as fp:
        loom = word_loom.load(fp)
        if build_prompt is not None:
            loom_sections = build_prompt.split(' ')
            num_loom_sections = len(loom_sections)
            if num_loom_sections not in [1, 2, 3]:
                raise click.BadParameter("Expected: 1-3 loom section names separated by space")
            elif num_loom_sections == 1:
                system_section = 'system_prompt'
                extra_context_section = 'context'
                question_section = loom_sections[0]
            elif num_loom_sections == 2:
                system_section = loom_sections[0]
                extra_context_section = 'context'
                question_section = loom_sections[1]
            else:
                system_section = loom_sections[0]
                extra_context_section = loom_sections[1]
                question_section = loom_sections[2]
        else:
            system_section = 'system_prompt'
            extra_context_section = 'context'
            question_section = 'question'
        question = loom[question_section]
        system = loom.get(system_section, '')
        if extra_context_section[0] == '[' and extra_context_section[-1] == ']':
            with open(extra_context_section[1:-1], 'r') as f:
                extra_context = f.read()
        else:
            extra_context = loom.get(extra_context_section, '')
        extra_context += '\n'
        marker_kwargs = {}
        if loom_markers:
            for loom_marker in loom_markers:
                marker, value = loom_marker.split('=')
                marker_kwargs[marker] = value
        question = question.format(**marker_kwargs)
        system = system.format(**marker_kwargs)
        if cot_source_path:
            with open(cot_source_path, 'r') as cot_content:
                chat = json.load(cot_content)
                chat.append({"role": "user",
                             "content": "\n".join([i for i in (system, extra_context, question) if i])})
                return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            return format(question, preamble=system, contexts=extra_context, delimiters=prompt_formatter.get_delimiters())


@click.command()
@click.option("--loom-file", help="An OgbujiPT word loom file to use for prompt construction")
@click.option("--loom-markers", help="Loom marker values", default=None, type=str, multiple=True)
@click.option('-p', '--prompt', default=None, type=str,
              help='Commandline prompt (overrides) prompt in YAML configuration')
@click.option('-t', '--temperature', default=1, type=float,
              help='Prompt generation temperature')
@click.option('-nt', '--num-tokens', default=-1, type=int,
              help='Overide number of tokens in config file')
@click.option('-f', '--prompt-format',
              type=click.Choice(PROMPT_FORMATS, case_sensitive=False))
@click.option('-a', '--adapter-path', default=None, type=str,
              help='Adapter to use instead of the one specified in the config file')
@click.option('-rp', '--repetition-penalty', default=0, type=float,
              help='The penalty factor for repeating tokens (none if not used)')
@click.option('--repetition-context-size', default=20, type=int,
              help='The number of tokens to consider for repetition penalty')
@click.option('-tp', '--top-p', default=CONFIG_DEFAULTS["top_p"], type=float,
              help='Sampling top-p')
@click.option('--min-p', default=-1, type=float, help='Sampling min-p')
@click.option('--min-p-tokens', default=1, type=int, help='Sampling min-p')
@click.option('--build-prompt', default=None, type=str,
              help='Which word loom sections to use in building the claim (space-separated list of sections)')
@click.option('--trust-remote-code/--no-trust-remote-code', default=False)
@click.option('--eos-token', default=None, type=str,
              help='End of sequence token for tokenizer')
@click.option('--seed', default=DEFAULT_SEED, type=int, help='PRNG seed')
@click.option("--cot-source", default=None,
              help="The name of the file with an apply chat template structure to use as the basis for a few-shot "
                   "prompt construction")
@click.argument('model_name')
def main(loom_file, loom_markers, prompt, temperature, num_tokens, prompt_format, adapter_path, repetition_penalty,
         repetition_context_size, top_p, min_p, min_p_tokens, build_prompt, trust_remote_code, eos_token, seed,
         cot_source, model_name):
    tokenizer_config = {}
    if eos_token is not None:
        tokenizer_config["eos_token"] = eos_token
    if trust_remote_code:
        tokenizer_config["trust_remote_code"] = True

    mx.random.seed(seed)

    model, tokenizer = load(
        model_name,
        adapter_path=adapter_path,
        tokenizer_config=tokenizer_config,
    )
    if loom_file:
        prompt = generate_prompt_from_loom(loom_file, loom_markers, get_prompt_formatter(prompt_format), build_prompt,
                                           cot_source, tokenizer)
    generate(
        model,
        tokenizer,
        prompt,
        num_tokens,
        verbose=True,
        temp=temperature,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        top_p=top_p,
        min_p=min_p,
        min_tokens_to_keep=min_p_tokens
    )

if __name__ == '__main__':
    main()
