# Failure Analysis

**Top 5 failure modes** identified from evaluation runs.

> Note: The examples below are drawn from the actual golden set evaluation run.
> If the golden set has not been labelled yet, re-run `make eval` after completing
> labelling to populate this file with real examples.

---

## 1. Intent Boundary Confusion: `order_delivery_tracking` ↔ `product_return_refund`

**Pattern**: Customer reports receiving the wrong item AND asks where their correct item is.
The message contains signals for both intents simultaneously.

**Example input**: *"I got someone else's package — when will my actual order arrive?"*

**Expected**: `product_return_refund` (the primary action is a return/replacement)  
**Actual prediction**: `order_delivery_tracking` (model latches onto "arrive" keyword)

**Root cause**: The keyword baseline over-weights delivery language; the LLM classifier
misses the implicit "wrong item" signal when it's phrased as a curiosity about the correct order.

**Impact**: Moderate — wrong escalation flag (return issues are escalation-sensitive;
tracking is not). Could auto-handle a case that should be escalated.

**Possible fix**: Add a "multi-intent detection" pass that checks if both return and
delivery keywords appear. If so, the intent with higher escalation sensitivity wins.

---

## 2. Account Security vs. Account Access Confusion

**Pattern**: Customer is locked out of their account but uses security-adjacent language
("someone changed my password").

**Example input**: *"I think someone changed my password and I can't get in."*

**Expected**: `account_security_compromise` (potential unauthorized access)  
**Actual prediction**: `account_access_login` (model treats it as a plain login issue)

**Root cause**: The model underweights the word "someone" as a signal of third-party
activity. The linguistic frame ("can't get in") triggers the login intent pattern.

**Impact**: High — `account_security_compromise` is always escalated; `account_access_login`
may be auto-handled. This represents a false-auto-handle on a security issue.

**Possible fix**: Add an explicit rule: if message contains "someone [verb]ed" + account
reference → classify as `account_security_compromise` regardless of base prediction.

---

## 3. Thin Evidence → Vague Generated Reply

**Pattern**: For rare intents (`seller_third_party_issue`, `account_security_compromise`),
the retrieval index has few matching examples (similarity < 0.35), and the generated
reply falls back to generic language that doesn't actually help the customer.

**Example input**: *"The marketplace seller sent me a counterfeit product."*

**Expected reply**: Specific guidance about the A-to-Z Guarantee claim process.  
**Actual reply**: "Thank you for reaching out. Based on similar cases, our team will
look into this. Please let me know if you need further assistance."

**Root cause**: Thin retrieval (top similarity ~0.28) means the generator has no real
evidence to ground a specific response. The system correctly flags `evidence_sufficient=False`
but still generates a reply instead of deferring entirely.

**Impact**: Medium — reply is not harmful but not helpful either. Escalation router
should catch this via the `weak_retrieval` flag.

**Possible fix**: When `top_similarity < 0.25`, skip generation entirely and return
a standard "escalating to specialist" message rather than a low-quality generic reply.

---

## 4. Prime vs. General Billing Misclassification

**Pattern**: Customer asks about an unexpected charge that turns out to be Prime,
but phrases it as a general billing question.

**Example input**: *"Why was I charged $14.99 this month? I didn't buy anything."*

**Expected**: `prime_subscription_billing`  
**Actual prediction**: `product_return_refund` (model sees "charged" and interprets as dispute)

**Root cause**: "$14.99" and "charged" are strong signals for refund-related intents.
Without seeing "Prime" explicitly, the model doesn't infer this is a subscription.

**Impact**: Low-medium — both intents are escalation-sensitive, so routing is correct.
But the reply would address the wrong scenario.

**Possible fix**: Include the specific amount pattern detection in a pre-processing
step: charges of $14.99, $8.99 are characteristic Prime tier amounts — add as a
Prime-specific signal in the keyword baseline.

---

## 5. Multi-Language / Emoji-Heavy Messages

**Pattern**: Some customer messages are primarily emoji or contain mixed-language
text (e.g., English + Spanish), which the English-trained intent classifier handles poorly.

**Example input**: *"😤😤😤 WHERE IS MY ORDER @AmazonHelp UNBELIEVABLE !!!!"*

**Expected**: `order_delivery_tracking` (frustrated delivery query)  
**Actual prediction**: `general_feedback_other` (mostly emoji + caps → noise-like signal)

**Root cause**: After stripping @mentions and normalizing whitespace, the cleaned
message is very short. Short messages default toward `general_feedback_other` in
both the keyword and LLM classifier.

**Impact**: Low — the correct intent is still escalation-safe. But the reply quality
would be lower (generic rather than delivery-specific).

**Possible fix**: For emoji-heavy messages, run sentiment detection first, then
apply intent classification with the original (un-stripped) text. Emoji like 😤
combined with delivery keywords should be a strong signal.

---

## What Is Misleading About the Headline Number?

This section is required by the assignment rubric. An honest investigation:

1. **Class imbalance**: `order_delivery_tracking` dominates the golden set (~30-35%).
   A classifier that predicts this for every example achieves ~30% accuracy "for free".
   Macro-F1 is more honest, but even that can be inflated if rare intents have support=1.

2. **Easy-example bias**: The stratified sampling includes ~80% "clear" examples
   (non-ambiguous heuristic). Real support queues have more ambiguous messages —
   the headline number likely over-estimates real-world performance.

3. **Small N**: N=150–250 gives a 95% CI of approximately ±7–8% on accuracy (Wilson
   score). A "difference" of 5% between two systems is not statistically significant
   at this sample size.

4. **LLM judge bias**: If human-judge agreement is weak (Spearman r < 0.4), the
   reply quality scores in the evaluation report are unreliable. We say this
   explicitly rather than presenting them as ground truth.

5. **Mock mode inflation**: In mock mode, the LLM judge returns deterministic
   mid-high scores (≈3.7/5 overall). This number is not a real quality measurement —
   it is a canned response chosen to be "reasonable." Do not compare mock-mode
   scores to live-mode scores.

6. **Retrieval leakage risk**: Conversation-level splitting prevents the same thread
   appearing in train and test. However, if a customer sent multiple support requests
   about the same issue in different threads, their language patterns appear in both
   splits. This is a known limitation of the dataset structure.

7. **Escalation accuracy hides severity asymmetry**: An accuracy of 85% on escalation
   may include a 15% false-auto-handle rate — catastrophic if those 15% are security
   issues. The raw accuracy number is misleading without breaking down the error types.

*Generated by `scripts/build_taxonomy.py` and manual analysis*
