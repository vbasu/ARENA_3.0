# Questions to come back to

Open questions I've parked mid-exercise, with enough context to pick each one up
cold. Newest first. When one is answered, move it to the Resolved section with a
one-line summary of the answer — the summary is the point, not the tick.

---

## Open

### Q9 — Bonus: what are heads 0.4 & 0.11 actually doing?
*Parked 2026-10-01, chapter 1 part 2. Deferred on ARENA's own advice, not
because I bounced off it.*

Heads `0.4` and `0.11` show up strongly in the ablation experiments (zero- and
mean-ablating them hurts the loss) but have *not* stood out in any of the other
analyses — they don't have a clean positional attention pattern, and they score
low on direct logit attribution. So something they do matters, and none of the
tools used so far can see it.

The task is to construct **my own targeted ablations** to work out what. ARENA's
hint: look at which source positions those heads attend to, and ask which source
positions are actually needed for the model to perform well. The implied method
is ablating a head's inputs *selectively by source offset* (e.g. keep only the
contribution from self-attention, or only from the token one position back, and
mean-ablate the rest), then watching the loss.

**Why I'm keeping this rather than skipping it:** it's one of the very few places
in the chapter that asks me to design the experiment instead of filling in a
function with a test attached. There's no assert telling me when I'm done, and
the notebook warns the results are ambiguous. That's the actual skill I want
([[designing-own-experiments-goal]]).

Entry point: cell ~116 of the 1.2 notebook, "Bonus - understand heads 0.4 & 0.11
(very hard!)". There is a "Partial answer (and some sample code)" dropdown —
**do not open it first.** Write down a hypothesis and an experiment before
looking at anything.

Also worth recording when I come back: which heads actually stood out in *my*
zero- and mean-ablation plots, since the bonus assumes 0.4 and 0.11 and I should
check that against what I measured.

Trigger to return: **after finishing the K-composition section** (1.2, section
4 — "Reverse-engineering circuits"). That section supplies the vocabulary for
reasoning about which source positions matter, which is exactly what the hint
asks for.

---

### Q8 — KV caching, and why it's valid
*Raised 2026-09-10, chapter 1 part 1. Skipped deliberately (5/5 difficulty,
1/5 importance), but it's the one skip worth returning to.*

I derived the causal-prefix invariant myself: position `i`'s representation is a
function of tokens `0..i` only, so two sequences sharing a `k`-token prefix have
bit-identical residual streams at every position below `k`. That invariant is
exactly *why* you can cache keys and values across generation steps instead of
recomputing the whole sequence per token.

Implement it and check: which tensors can be reused, which must be recomputed,
and what breaks if the prefix invariant were violated. Cell ~180 of the part 1
notebook is the entry point.

---

### Q7 — My causal mask depended on the score values, not the positions
*Raised 2026-09-08, chapter 1 part 1. Set aside rather than fixed.*

I wrote `mask = t.triu(attn_scores, diagonal=1).bool()`, which takes the upper
triangle of the *scores themselves* and treats nonzero as True. So a score that
happens to be exactly 0.0 never gets masked, and an all-zero score matrix gets
no mask at all. It passed `test_causal_mask` because random scores are never
exactly zero. I replaced it with the provided solution instead of fixing mine.

Come back and fix my own version. Then check it against the reference-free
property test: set `W_Q = W_K = 0` so every score is zero, and confirm the
pattern is exactly uniform over allowed positions — row `i` should be
`1/(i+1)` repeated `i+1` times, then zeros.

---

### Q6 — Why the first token is dropped from loss and accuracy
*Raised 2026-09-08, chapter 1 part 1. I said "not sure I'm following, let's come
back to that."*

We compare `logits[:, :-1]` against `tokens[:, 1:]`. I understood dropping the
last logit (no target for it). I did not follow dropping the first token.

The claim to work through: `seq` positions emit `seq` logit vectors, and
position `i`'s vector predicts token `i+1` — so the targets covered are
`1..seq`, and token 0 is only ever an input, never predicted by anything. Then:
the dataset is built with `add_bos_token=True`, so token 0 is `<|endoftext|>`
(50256), the same id GPT-2 uses for EOS and padding. So the token being dropped
isn't a word at all, and position 0's actual job is predicting the first *real*
token.

Re-derive the slot-counting argument, then work out what would change if BOS
were *not* prepended.

---

### Q5 — The gradient depends on your choice of inner product
*Raised 2026-08-26, from the `dL = <dL/dy, dy>` discussion.*

The derivative of a scalar `L` at a point is a **linear map** `R^n -> R`. The
gradient is the *vector representing* that map via an inner product — so it
exists only once an inner product is chosen.

Swap the standard dot product for `<u, v>_M = u^T M v`, with `M` symmetric
positive-definite. The derivative map `DL` is unchanged: it's the same function
of `dy`. What happens to the gradient *vector* that represents it?

Then: which optimizer from chapter 0 part 3 is quietly doing something of that
shape?

---

### Q4 — Adjoint of a gather
*Raised 2026-08-26, underwrites transposed convolutions.*

Let `G` be a gather: `(Gx)_i = x_{idx[i]}`, pulling entries out of `x`, possibly
the same entry more than once. Push `<Gx, y>` through the adjoint identity
`<Ax, y> = <x, A*y>` until `x` sits alone on the left of the comma, and read off
what `G*` must be.

Then check the answer against why part 4's `getitem` backward needed
`np.add.at` rather than plain assignment.

(Status 2026-09-10: substantially answered in passing while doing `Embed`.
Embedding is multiplication by a one-hot matrix `P`, so the backward is `P^T`,
and if a token appears twice in a sequence `P^T` has a row with two ones in it
— which sums the incoming gradients. Scatter-add, not scatter-assign. I have
not done the `<Gx, y> = <x, G*y>` derivation myself, which was the actual
point of parking it.)

---

### Q3 — Output size of a transposed convolution
*Raised 2026-08-26, chapter 0 part 5.*

From the stamp picture alone — `i` stamps, each `k` wide, spaced `s` apart —
write the output length as a function of `i`, `k`, `s`. Check it against both
worked cases (`i=2, k=3, s=1 -> 4` and `i=2, k=3, s=2 -> 5`).

Then predict what padding does to it, given that padding on a transposed conv
*removes* output rather than adding it.

(Status 2026-08-31: I finished part 5 without doing the transposed-conv
implementation exercises, so this is still genuinely open. Cells ~95-107 of the
part 5 notebook are the entry point.)

---

### Q2 — Why the KL term penalizes the mean, not just the variance
*Raised 2026-08-26, chapter 0 part 5 (VAEs).*

The KL is between `q(z|x) = N(mu, sigma^2)` and the prior `N(0, I)`, so it
pushes against both `sigma != 1` and `mu != 0`.

Suppose you kept only the `sigma` part. What stops the encoder from scaling
every `mu` up by a factor of 1000 while leaving `sigma = 1` — and what has it
effectively rebuilt if it does?

---

### Q1 — The trace / differential method for matrix calculus
*Deferred 2026-08-25, chapter 0 part 4.*

`dL = <dL/dx, dx>`, rotate under the trace, read off the gradient. I followed
the explanation but couldn't apply it unaided, and parked it deliberately.

Agreed trigger to return: deriving **BatchNorm or LayerNorm backward by hand**,
where mean and variance depend on every element and the index-notation version
gets ugly. When I do come back, worked examples rather than re-reading:
`L = <a, x b>`, then `L = tr(x^T x)`, then `x @ y` from scratch.

---

## Resolved

*(nothing yet)*
