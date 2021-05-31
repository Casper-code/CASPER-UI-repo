import torch
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizer


def load_model(output_dir):
    # Load the BERT.
    print('Loading XLMRobertaForSequenceClassification...')
    model = XLMRobertaForSequenceClassification.from_pretrained(output_dir)
    tokenizer = XLMRobertaTokenizer.from_pretrained(output_dir)
    return model, tokenizer


def predict(model, tokenizer, sentence):
    encoded_dict = tokenizer.encode_plus(
        sentence,
        truncation=True,
        add_special_tokens=True,
        padding='longest',
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    input_id = torch.LongTensor(input_id)
    attention_mask = torch.LongTensor(attention_mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loaded = model.to(device)
    input_id = input_id.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model_loaded(input_id, token_type_ids=None, attention_mask=attention_mask)
    logits = outputs[0]
    index = logits.argmax()

    if index == 0:
        return False
    return True
