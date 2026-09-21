You are reading one answer from a research interview and recording what it yielded.

The goal this question was serving:
{goal}

Question asked:
{question}

Answer given:
{answer}

**First, extract the facts.** Go through the answer clause by clause and pull out every
specific thing stated: an action taken, a duration, a count, a date, a tool named, a
decision made, a consequence. Write each as a short standalone sentence in the
respondent's own terms. Do not interpret, summarise, or combine them.

Worked example. Answer: "It was two days before the deadline, and I was trying to find the
file. I checked the folders, the cloud storage, everything. I spent about forty minutes
before I gave up."

facts:
- The deadline was two days away when the file went missing.
- She checked the folders and the cloud storage.
- She searched for about forty minutes before giving up.

An answer you are about to classify as concrete must produce at least one fact. If you
cannot extract one, the answer was not concrete — classify it as vague instead.

Then classify the answer:
- concrete: describes a specific episode, or gives a number, duration, name, or sequence
  of events that can be checked against later answers.
- vague: on topic but unquantified or unanchored. "A while", "pretty often", "it was
  annoying" with no occasion attached.
- evasive: avoids the question, redirects, or answers a different question.
- new_thread: introduces a topic that was not asked about and may matter more than the
  question was reaching for.

Set goal_progress to how well this goal now stands:
- covered: you could write up a finding about this goal from what you now have.
- shallow: the respondent engaged but the material is too thin to write up.
- untouched: the answer gave nothing toward this goal.

If the answer introduced an unplanned topic that bears on the research, name it in
emergent_topic. Otherwise leave it null.
