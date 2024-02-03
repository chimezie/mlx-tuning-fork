from ogbujipt.prompting import pdelim, format

#https://github.com/OoriData/OgbujiPT/pull/70
MISTRAL_INSTRUCTION_DELIMITERS_NO_BOS = {
    pdelim.FIXED_PREAMBLE: '[INST]',
    pdelim.POSTQUERY: '\n[/INST]',
}


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["input"], delimiters=MISTRAL_INSTRUCTION_DELIMITERS_NO_BOS)

    @classmethod
    def get_output(cls, record) -> str:
        return record["output"]
