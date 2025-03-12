1. What should you do if the two models have different tokenizers?
   It may be difficult to use contrastive decoding when the two models have different tokenizers. With the token-level decoding algorithm, we subtract the probabilities for tokens that match up exactly. If the tokens are different, we can't subtract the probabilities and this reduces the effectiveness of contrastive decoding.
   We would need to convert one tokenizer to the other or create some sort of mapping between the tokens. However, re-tokenizing or training a model to learn how to map one model's tokenization to the other may take extra training. If one tokenizer produces more fine-grained subwords than the other, we could try to sum the log probabilities of the subwords that correspond with a single token in the other model. For instance if one tokenizer breaks "computer" into "com", "pu" and "ter" we can try to generating more tokens until it sums up and adding together the log probabilities of the smaller tokens before subtracting. But it is not guaranteed that we can generate tokens that exactly add up.

2. Do you think contrastive decoding is used in practice?

Based on the results of contrastive decoding for this assignment it seems using contrastive decoding is dependent on the specific task. The sample prompt asks for a docstring for a function that updates ELO scores. Specifically, the prompt asks for a “very very brief doctstring”. I generated an output with the amateur model, the expert model, and then using my contrastive decoding function to compare the results. The amateur model returned the entire function with the doctstring added:Amateur model output:
/\*\*

- Updates the Elo scores of a list of players based on their performance in a game.
- The Elo score is calculated using the formula:
- E = K \* (S1 - S2) / (1 + sqrt(S1^2 + S2^2))
- where K is a constant factor, S1 and S2 are the scores of the two players,
- and E is the updated score.
-
- @param {Array} scores - An array of scores for each player.
- @param {Array} results - An array of results, where each result is an object with 'first' and 'second' keys,
-                           and 'outcome' key indicating the outcome of the game.
- @param {number} kFactor - A constant factor to adjust the Elo score.
- @returns {Array} - An array of updated scores for each player.
  \*/
  function updateEloScores(scores, results, kFactor = 4) {
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

  }

```<|im_end|>

The expert model returned a text description of what the function does:
This function updates Elo scores based on the results of a series of games. It takes in an object `scores` representing the current Elo scores of each player, an array `results` containing the outcomes of each game, and an optional `kFactor` parameter that determines the strength of the update. The function iterates over each game result, calculates the expected score for each player based on the Elo rating system, and then updates the Elo score of each player based on the outcome of the game. The function returns the updated Elo scores.<|im_end|>

Finally the contrastive model also generated a description of what the function does, rather than a docstring like the amateur model. I tested for varying levels of alpha (0.1, 0.3, 0.5, 0.7, 0.9), being more aggressive when subtracting amateur model probabilities. The outputs were all of similar style but varying lengths. It is interesting to note that increasing alpha was proportional to a shorter output. For alpha >= 0.5, the contrastive decoding approach generated shorter outputs than the expert model. Below is the output with alpha=0.7:
This function updates Elo scores based on game results. It takes in a dictionary of scores, a list of game results, and an optional k-factor (defaulting to 4). For each game result, it calculates the expected score of each player based on their current Elo score. It then adjusts their Elo score based on the outcome of the game (1 for a win, -1 for a loss, 0 for a draw) using the Elo rating system.<|im_end|>
I would argue for a task like this sample prompt, it’s not clear that a contrastive decoding approach with the token level algorithm is preferred. It generates a slightly briefer description than the expert model but not by much. If the desired output was a docstring it’s possible that the amateur’s model is actually most preferred.

The paper shows that contrastive decoding can achieve better performance and more efficient strategy for getting better results than methods like finetuning. The example in reducing redundancy in the larger model's output shows clear performance gains. However, using contrastive decoding in this task show some drawbacks to using contrastive decoding in practice. It can be difficult to find the right “parameters” to generate objectively better outputs for a given task. For instance you may have to experiemnt with different levels of alpha and maybe use beam search algorithm instead of token level algorithm. It seems difficult to estimate how much contrastive decoding can increase performance and benefits may not outweigh the additional compute involved for generating with a smaller model or time needed to find the desired contrastive decoding output. These drawbacks could mean contrastive decoding isn't really used in practice.
```
