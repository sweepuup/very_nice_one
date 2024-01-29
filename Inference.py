import torch
import Train
import Tokenizer


def load_model(model_path, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
    model = Train.Transformer(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def prepare_input(input_text):
    input_tokens = Tokenizer.tokenize_sequence(input_text)
    # input_tokens = Tokenizer.pad_to_length(input_tokens, Train.max_sequence_length)
    input_ids = torch.tensor(input_tokens).unsqueeze(0)  # Add batch dimension
    return input_ids


# def generate_output(model, input_ids):
#     with torch.no_grad():
#         output_logits = model(input_ids)
#     predicted_token_ids = torch.argmax(output_logits, dim=-1)
#     output_text = Tokenizer.detokenize_sequence(predicted_token_ids[0].tolist())
#     return output_text


# def generate_output(model, input_ids, max_length, eos_token=Tokenizer.vocabulary.get('<EOS>')):
#     with torch.no_grad():  # No need to track gradients during inference
#         input_tensor = torch.tensor(input_ids)
#         output_seq = []
#
#         for i in range(50):
#             output = model.generate(input_tensor)
#             print(f'output.size(): {output.size()}')
#             next_token = torch.argmax(output[0, i, :], dim=-1).item()  # Take last token from sequence
#             output_seq.append(next_token)
#             if next_token == eos_token:
#                 break  # Stop if EOS token is generated
#             next_token_tensor = torch.tensor([[next_token]]).to(Train.device)  # Convert and move to device
#             input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)  # Concatenate
#         print(f'Generated tokens: {output_seq}')
#         AI_response = Tokenizer.detokenize_sequence(output_seq)
#         return AI_response


def generate_output(model, input_ids, max_length, eos_token=Tokenizer.vocabulary.get('<EOS>')):
    out = model.generate(input_ids)
    preds = torch.argmax(out, dim=-1)
    output_tokens = []
    for token in preds[0]:
        output_tokens.append(token.item())
    AI_response = Tokenizer.detokenize_sequence(output_tokens)
    return AI_response


# Example usage
model_path = 'models/my_model.pt'
input_text = ''

model = load_model(model_path, Train.d_model, Train.ffn_hidden, Train.num_heads, Train.drop_prob, Train.num_layers)
model.to(Train.device)
input_ids = prepare_input(input_text)
output_text = generate_output(model, input_ids.to(Train.device), Train.max_sequence_length)

print("Generated Output:", output_text)