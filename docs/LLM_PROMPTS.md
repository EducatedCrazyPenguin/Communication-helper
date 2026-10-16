# LLM prompts

Separate versioned prompts cover topics, critic, listener, scores, question generation, answer resolution, retry evaluation, requested examples and profile wording. Inputs quote transcript as untrusted data. Output schemas and exact span checks are mandatory; one bounded repair is permitted, then a durable failure. No tool use, assumed facts, diagnoses or subject-knowledge grading.

Listener receives only transcript and its own cited ledger, never topic answers/prep/critic/profile. Retry analysis is independent of the first attempt and cross-exam facts. Model answer is gated until the learner retries. Provider/model/prompt/rubric/config hashes are retained with successful analysis.
