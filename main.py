import transformers as tr
import torch

# Set up models & tokenizers
amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path, torch_dtype=torch.float16).to(device)
expert = tr.AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=torch.float16).to(device)

# Define prompt
user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

def contrastive_generation(amateur, expert, prompt, max_tokens=1000, a=0.5):
    """
    Generates text using contrastive decoding between amateur and expert models.

    Args:
        amateur (torch.nn.Module): Amateur language model.
        expert (torch.nn.Module): Expert language model.
        prompt (str): Input text prompt.
        max_tokens (int): Maximum number of tokens to generate (default: 1000).
        a (float): Scaling factor for amateur influence (0 to 1, default: 0.5).

    Returns:
        str: Generated text response.
    """
    if not (0 <= a <= 1):
        raise ValueError("Parameter 'a' must be between 0 and 1.")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eos_token_id = tokenizer.eos_token_id

    output_tokens = []
    current_input = input_ids

    with torch.no_grad():
        for _ in range(max_tokens):
            amateur_logits = amateur(current_input).logits[:, -1]
            expert_logits = expert(current_input).logits[:, -1]

            amateur_probs = torch.softmax(amateur_logits, dim=-1)
            expert_probs = torch.softmax(expert_logits, dim=-1)

            contrast_probs = expert_probs - a * amateur_probs
            contrast_probs = torch.clamp(contrast_probs, min=1e-9)

            next_token_id = contrast_probs.argmax(dim=-1).item()
            output_tokens.append(next_token_id)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            current_input = torch.cat([current_input, torch.tensor([[next_token_id]], device=device)], dim=1)

    return tokenizer.decode(output_tokens, skip_special_tokens=True)

print('Contrastive decoding output:')
print(contrastive_generation(amateur, expert, prompt))
