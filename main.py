import transformers as tr
import torch

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print('loading tokenizer')
tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
print('loading amateur')
amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
print('loading expert')
expert = tr.AutoModelForCausalLM.from_pretrained(expert_path)

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

def regular_generation(model, prompt, max_tokens=1000):
    input = tokenizer(prompt, return_tensors="pt").input_ids
    input = input.to(amateur.device)

    output_tokens = []
    current_input = input

    for _ in range(max_tokens):
        probs = model(current_input, return_dict=True).logits[:,-1].softmax(dim=-1)

        next_token_id = probs.argmax()
        output_tokens.append(next_token_id)
        current_input = torch.cat([current_input, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

        if next_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens)

def contrastive_generation(amateur, expert, prompt, max_tokens=1000):
    a = 0.9 #(expert - α * amateur)
    input = tokenizer(prompt, return_tensors="pt").input_ids
    input = input.to(amateur.device)
    
    output_tokens = []
    current_input = input
    
    for _ in range(max_tokens):
        amateur_probs = amateur(current_input).logits[:,-1].softmax(dim=-1)
        expert_probs = expert(current_input).logits[:,-1].softmax(dim=-1)

        contrast_probs = expert_probs - a*amateur_probs

        next_token = contrast_probs.argmax()
        output_tokens.append(next_token)
        
        current_input = torch.cat([current_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(output_tokens)

print('contrastive: ')
print(contrastive_generation(amateur, expert, prompt))
# print('expert only: ')
# print(regular_generation(expert, prompt))
# print('amateur only: ')
# print(regular_generation(amateur, prompt))
