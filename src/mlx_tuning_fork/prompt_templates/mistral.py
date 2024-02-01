from ogbujipt.prompting import pdelim, format, MISTRAL_INSTRUCTION_DELIMITERS_NO_BOS


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["input"], delimiters=MISTRAL_INSTRUCTION_DELIMITERS_NO_BOS)

    @classmethod
    def get_output(cls, record) -> str:
        return record["output"]
