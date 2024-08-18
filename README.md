# MistralGuard: Long Context Safeguards for Human-AI Conversations

by Giosuè Castellano, Giovanni Gatti, Rolf Minardi, Fabio Pernisi.

Final Research Project for the class _20879 - Language Technology_ @ Bocconi University (2024).

Full Paper: [MistralGuard: Long Context Safeguards for Human-AI Conversations](assets/MistralGuard.pdf).

## Abstract
> As Large Language Models (LLMs) are increasingly being adopted and integrated into
> a broad range of applications, it becomes essential to assess and ensure their safety. 
> Recent research has demonstrated that these models can be maliciously pushed to operate outside
> their guidelines (i.e., jailbroken) using numerous demonstrations of undesired behavior, 
> exploiting their long-context capabilities. While existing solutions like Llama Guard have 
> addressed these issues, they remain constrained by the token limits of the models they use and
> often lack precision in detecting diverse jailbreaking attempts in longer contexts. This work
> introduces MistralGuard, an extended version of the Mistral 7B Model, specifically designed
> to classify and mitigate jailbreaking prompts up to 32,000 tokens. We leverage a novel dataset
> derived from engaging a non-aligned model with unsafe inputs. In comparative evaluations,
> MistralGuard outperforms the state-of-the-art model GPT-4o and the first iteration of Llama
> Guard across several contexts. It demonstrates superior performance in zero-shot (no-context),
> many-shot (short-context under 8k tokens), and long-context scenarios (8k to 32k tokens).

<div align="center">
  <img src="assets/intro.png" alt="img" width="400" height="auto">
  <p><em>Example of a long context prompt.</em></p>
</div>

## Results
Evaluation results can be reproduced running the `evaluation` [notebook](src/evaluation.ipynb).

## Contact Information
Contact the authors at: name DOT surname AT studbocconi DOT it.

