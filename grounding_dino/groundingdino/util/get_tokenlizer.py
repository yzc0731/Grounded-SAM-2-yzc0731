from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

def get_tokenlizer(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        # bert_base_uncased_path = "../../../bert"
        project_name = 'Grounded-SAM-2'
        parts = os.path.abspath(__file__).split(os.sep)
        project_index = parts.index(project_name)
        project_root = os.sep.join(parts[:project_index + 1])
        bert_base_uncased_path = os.path.join(project_root, "bert")
        # print("bert_base_uncased_path: {}".format(bert_base_uncased_path))

        return AutoTokenizer.from_pretrained(bert_base_uncased_path)

    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        project_name = 'Grounded-SAM-2'
        parts = os.path.abspath(__file__).split(os.sep)
        project_index = parts.index(project_name)
        project_root = os.sep.join(parts[:project_index + 1])
        bert_base_uncased_path = os.path.join(project_root, "bert")
        # print("bert_base_uncased_path: {}".format(bert_base_uncased_path))

        return BertModel.from_pretrained(bert_base_uncased_path)

    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
