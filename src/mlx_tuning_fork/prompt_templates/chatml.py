from ogbujipt.prompting import format, CHATML_DELIMITERS


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["input"], delimiters=CHATML_DELIMITERS)

    @classmethod
    def get_output(cls, record) -> str:
        return record["output"]
