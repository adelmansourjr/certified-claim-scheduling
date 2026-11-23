# What an exact check schedule does and does not guarantee

## Objective

A source document has been given, as have the atomic facts to check against it. Let x_i be the fixed binary result of checking subclaim i. We model the target as AND(x_1, …, x_n). The scheduler may compute this exactly by reading checks in batches and terminating after observing a failed check. The output must be correct for every possible outcome vector. If an estimated prior assigns a zero probability to an outcome, that prior is still insufficient as a certificate. This is why the code never infers a negative or positive subclaim from correlations.

A schedule consists of an ordered partition into nonempty batches of at most k items. The only path on which work continues is all checks seen so far passing. Thus observing their individual labels gives no additional branches among continuing states. A deterministic scheduler induces one ordered partition for that path. Failure paths end immediately. Randomizing over schedules cannot improve expected cost for a fixed known distribution.

Let c(B) be the batch overhead plus the sum of item costs. Let q(S) be the probability that every check in set S passes. For any continuation starting after S, the cost of its next batch contributes q(S)c(B) before the look reveals its outcome. Afterward the unconditioned minimum remaining contribution is F(S∪B). Therefore
```
F(S) = min_{B feasible, disjoint from S} [q(S)c(B) + F(S union B)].
F(all) = 0.
```
By backward induction this minimizes prior-weighted expected cost, and using q(S) directly avoids dividing by small survival probabilities. The state count is 2^n. The feasible transitions out of S number at most the sum of binomial coefficients up to the capacity over its remaining items. Classical stochastic Boolean evaluation and batched series-system testing motivate this exact calculation. [Deshpande, Hellerstein and Kletenik](https://arxiv.org/abs/1303.0726) · [Daldal et al.](https://doi.org/10.1007/s10479-016-2345-7)

## Same marginals and different optimal first checks

Consider independent fair bits a and b. Define one law by x=(a,1−a,b,b), and a second law by exchanging the two pairs. In both laws, mix these assignments with probability ε of four independent fair bits, for 0 < ε < 1/2. This ensures full support on all possible label vectors and keeps every marginal at 1/2. Each check costs 1, each round costs 2, and capacity is 2.

Under the first law, start with the opposite pair. It costs 4 to check and passes only with probability ε/4. In that rare case the other two outcomes are independent fair bits, and testing both together costs 4, better than the conditional cost 4.5 of testing them sequentially. Total expected cost is 4 + ε. The second law swaps which pair is opposite. Any singleton first costs 3 and has at least one further check costing at least 3 with probability 1/2, making it cost at least 4.5. Any other first pair has pass probability at least 1/4 and must do at least one further check in that event, making it cost at least 4.75. Thus each law has a different uniquely optimal first batch.

For a scheduler given only the shared marginals and equal prior weight on the two candidate laws, expected risk equals its risk under their equal mixture. In this mixture, every pair passes with probability 1/4 and every triple with probability 1/8. Starting with any pair and checking both remaining items together gives cost 4 + (1/4)4 = 5. Checking those remaining items sequentially has higher conditional cost 4.5. A singleton first costs 3. Following it with a pair already costs 3 + (1/2)4 = 5 before any final check. Following it with a singleton needs a further check with probability 1/4 and costs at least 3 + (1/2)3 + (1/4)3 = 5.25. The optimal common schedule therefore costs 5. Its average regret relative to the law-specific schedules is exactly 1−ε, which is 0.9 at ε=0.1. This bound applies to randomized schedules by linearity and is achievable by always evaluating the two batches of size two.

The mathematical example establishes possible cost sensitivity to dependence under this checking interface. The WiCE data do not validate the synthetic dependence pattern. In fact, measured cost difference between the two learned DP variants is inconclusive in the primary setting.

## What is being costed

The cost policy charges 20 plus claim word count for each observed annotation and a fixed or document-length overhead per batch. A zero-overhead case approximates completely reusable context. Fixed overhead cases represent increasing repeated setup costs. The longest case charges the entire evidence's word count repeatedly. This is a trade-off experiment in specified cost units. It does not measure an LLM's tokens, batching behaviour, calibration, latency or memory usage. The lexical ranker does see the full document once to estimate the priors. Measured whole-process CPU resources include that work and policy construction, but the proxy differences do not subtract or convert the measured CPU costs of invoking one schedule rather than another.

The all-pass target corresponds to full support for every supplied decomposition, using annotations that are already available. A decomposition can omit a needed claim, a human label can be imperfect, and a future verifier can be context dependent or alter its answers when asked to batch. Those are outside the exact stopping guarantee. Follow-up with a real checker should measure input and generated tokens, document prefills reused, peak memory and answer changes at a fixed accuracy threshold, stratifying by decomposition length and document length.
