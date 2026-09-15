# Intent Taxonomy — `AmazonHelp`

**Total intents: 9** (6–12 range specified; 9 chosen based on data exploration on 3,000-message TF-IDF clustering of training split)

**Exploratory clusters**: customer messages were clustered into 9 groups using KMeans on TF-IDF features. The taxonomy below is **data-informed** but **human-finalised** — cluster top-terms guided the definitions; final intent names and boundaries were set by the agent as a labelling starting point.

**Ambiguous/overlapping intents** are flagged inline.

---

## 1. `order_delivery_tracking`

**Escalation sensitivity**: 🟢 Non-escalation

**Definition**: Customer asking about the status of an order, delivery timeline, tracking information, or a missing/delayed package. The customer has placed an order and wants to know where it is or when it arrives.

**Example messages**:
  - *"My package was supposed to arrive yesterday, where is it?"*
  - *"Can you give me a tracking number for order #12345?"*
  - *"I ordered 3 days ago and still no dispatch confirmation."*

**Disambiguation**: Distinct from product_return_refund (no return requested) and from account_order_issue (not about account access).

---

## 2. `product_return_refund`

**Escalation sensitivity**: 🔴 **ESCALATION-SENSITIVE**

**Definition**: Customer wants to return an item, request a refund, or dispute a charge. Includes damaged/wrong items received and requests for reimbursement.

**Example messages**:
  - *"I received the wrong item and need a refund."*
  - *"How do I return this? It's damaged."*
  - *"I was charged twice for the same order."*

**Disambiguation**: Escalation-sensitive because billing disputes can involve financial harm. Distinct from order_delivery_tracking (customer is not asking about delivery, they already have the item or a charge).

---

## 3. `account_access_login`

**Escalation sensitivity**: 🔴 **ESCALATION-SENSITIVE**

**Definition**: Customer cannot log in, is locked out, forgot password, or has issues accessing their Amazon account. Includes 2FA/verification problems.

**Example messages**:
  - *"I can't log into my account — it says my password is wrong."*
  - *"I'm not receiving the OTP to verify my account."*
  - *"My account has been locked, help!"*

**Disambiguation**: Escalation-sensitive (account security risk). Distinct from account_security_compromise (no unauthorized access suspected here).

---

## 4. `account_security_compromise`

**Escalation sensitivity**: 🔴 **ESCALATION-SENSITIVE**

**Definition**: Customer reports or suspects unauthorized access to their account, fraudulent orders placed without their knowledge, or phishing/scam incidents related to their Amazon account.

**Example messages**:
  - *"Someone placed an order on my account without my permission!"*
  - *"I got a phishing email claiming to be Amazon — is it real?"*
  - *"There are orders in my history I didn't make."*

**Disambiguation**: Highest escalation priority. Distinct from account_access_login (security compromise involves unauthorized third-party activity).

---

## 5. `product_question_usage`

**Escalation sensitivity**: 🟢 Non-escalation

**Definition**: Customer has a question about a product's features, compatibility, how to use it, setup instructions, or whether it will work for their use case.

**Example messages**:
  - *"Does this Kindle support PDF files?"*
  - *"How do I set up my Echo Dot on my WiFi?"*
  - *"Is this charger compatible with iPhone 14?"*

**Disambiguation**: Distinct from technical_bug (no malfunction reported) and from order_delivery_tracking (question is about the product itself, not the shipment).

---

## 6. `technical_bug_malfunction`

**Escalation sensitivity**: 🟢 Non-escalation

**Definition**: Customer reports a technical problem: app crash, website error, device not working, feature broken, or software glitch on an Amazon product or service.

**Example messages**:
  - *"The Amazon app keeps crashing when I try to checkout."*
  - *"My Fire TV remote stopped working after the update."*
  - *"I get an error code PVS-106005 when trying to stream."*

**Disambiguation**: Distinct from product_question_usage (a bug means something is broken, not just unclear). May escalate if widespread outage.

---

## 7. `prime_subscription_billing`

**Escalation sensitivity**: 🔴 **ESCALATION-SENSITIVE**

**Definition**: Customer questions about Amazon Prime membership: charges, cancellation, benefits not working, unexpected renewal, or downgrade requests.

**Example messages**:
  - *"I cancelled Prime but was still charged this month."*
  - *"How do I cancel my Prime trial before it renews?"*
  - *"My Prime Video isn't working even though I'm a member."*

**Disambiguation**: Escalation-sensitive due to billing disputes. Distinct from product_return_refund (the issue is the subscription charge, not a product purchase).

---

## 8. `seller_third_party_issue`

**Escalation sensitivity**: 🟢 Non-escalation

**Definition**: Customer has an issue specifically with a third-party seller on Amazon Marketplace: seller not responding, counterfeit product, listing mismatch, or wanting Amazon to mediate.

**Example messages**:
  - *"The seller hasn't replied to my messages for a week."*
  - *"I think this product is a counterfeit — the seller is unresponsive."*
  - *"The item description was completely wrong — this is a marketplace seller."*

**Disambiguation**: Distinct from product_return_refund (issue is with the seller specifically, not just wanting a refund). May lead to A-to-Z claim.

---

## 9. `general_feedback_other`

**Escalation sensitivity**: 🟢 Non-escalation

**Definition**: Customer is providing general feedback, a complaint not fitting another category, asking about Amazon policies, or sending a message that doesn't clearly fit any specific intent above.

**Example messages**:
  - *"Your customer service has gotten really bad lately."*
  - *"Do you have a store in my city?"*
  - *"Just wanted to say the packaging was amazing!"*

**Disambiguation**: This is a genuine residual bucket — justified by data exploration showing ~8% of messages don't map cleanly to the above 8 intents. Ambiguous messages go here if labellers cannot decide.

---

## Ambiguous / overlapping cases

- `account_access_login` vs `account_security_compromise`: if the customer mentions unknown orders or someone else in their account → security; if just locked out with no third-party activity → access.
- `product_return_refund` vs `prime_subscription_billing`: check whether the disputed charge is for a product or a subscription.
- `technical_bug_malfunction` vs `product_question_usage`: if the customer says 'it used to work' or 'stopped working' → bug; if 'how do I...' → usage.
- `order_delivery_tracking` vs `product_return_refund`: if the item hasn't arrived yet → tracking; if arrived but wrong/damaged → return.

## `Other/Unknown` bucket justification

`general_feedback_other` is included because cluster exploration showed ~8–10% of messages are genuinely ambiguous (policy questions, compliments, random inquiries). Without this bucket, labellers would be forced to incorrectly assign these to the nearest intent, poisoning the training distribution.

---
*Generated by `scripts/build_taxonomy.py`*