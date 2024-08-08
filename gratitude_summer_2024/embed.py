import torch


@torch.no_grad()
def mean_pool_sentence_embedding(sentences, tok, m, batch_size, device, max_seq_len):
    """
    Code modified from the sentence-transformers library and huggingface docs: https://sbert.net/
    """

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):

        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # Collect embeddings
    all_embbeddings = []
    for i in range(0, len(sentences), batch_size):

        # Tokenize sentences
        encoded_input = tok(
            sentences[i : i + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_seq_len,
        ).to(device)

        model_output = m(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        all_embbeddings += [sentence_embeddings]

    sentence_embeddings = torch.cat(all_embbeddings, axis=0)

    return sentence_embeddings.cpu().data.numpy()