import click
from mlx_tuning_fork.config import CONFIG_DEFAULTS, get_prompt_formatter, PROMPT_FORMATS
from ogbujipt import word_loom
from mlx_lm.utils import load, generate
from mlx_lm.generate import colorprint_by_t0

DEFAULT_SEED = 0


def generate_prompt_from_loom(loom_file, loom_markers, prompt_formatter, build_prompt):
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
        if loom_markers is not None:
            marker, value = loom_markers.split('=')
            question = question.format(**{marker: value})
        return format(question, preamble=system, contexts=extra_context, delimiters=prompt_formatter.get_delimiters())


@click.command()
@click.option("--loom-file", help="An OgbujiPT word loom file to use for prompt construction")
@click.option("--loom-markers", help="Loom marker values", default=None, type=str)
@click.option('-p', '--prompt', default=None, type=str,
              help='Commandline prompt (overrides) prompt in YAML configuration')
@click.option('-t', '--temperature', default=None, type=float,
              help='Prompt generation temperature')
@click.option('-nt', '--num-tokens', default=-1, type=int,
              help='Overide number of tokens in config file')
@click.option('-f', '--prompt-format',
              type=click.Choice(PROMPT_FORMATS, case_sensitive=False))
@click.option('-a', '--adapter', default=None, type=str,
              help='Adapter to use instead of the one specified in the config file')
@click.option('-rp', '--repetition-penalty', default=0, type=float,
              help='The penalty factor for repeating tokens (none if not used)')
@click.option('--repetition-context-size', default=20, type=int,
              help='The number of tokens to consider for repetition penalty')
@click.option('-tp', '--top-p', default=CONFIG_DEFAULTS["top_p"], type=float,
              help='Sampling top-p')
@click.option('--build-prompt', default=None, type=str,
              help='Which word loom sections to use in building the claim (space-separated list of sections)')
@click.option('--trust-remote-code/--no-trust-remote-code', default=False)
@click.option('--eos-token', default=None, type=str,
              help='End of sequence token for tokenizer')
@click.option('--seed', default=DEFAULT_SEED, type=int, help='PRNG seed')
@click.option("--colorize/--no-colorize", default=False, help="Colorize output based on token probability")
@click.argument('model')
def main(loom_file, loom_markers, prompt, temperature, num_tokens, prompt_format, adapter, repetition_penalty,
         repetition_context_size, top_p, build_prompt, trust_remote_code, eos_token, seed, colorize, model_name):
    if loom_file:
        prompt = generate_prompt_from_loom(loom_file, loom_markers, get_prompt_formatter(prompt_format), build_prompt)
    tokenizer_config = {}
    if eos_token is not None:
        tokenizer_config["eos_token"] = eos_token

    formatter = colorprint_by_t0 if colorize else None

    model, tokenizer = load(
        model_name,
        adapter_path=adapter,
        tokenizer_config=tokenizer_config,
    )

    generate(
        model,
        tokenizer,
        prompt,
        num_tokens,
        verbose=True,
        formatter=formatter,
        temp=temperature,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        top_p=top_p,
    )


if __name__ == '__main__':
    main()
