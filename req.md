# Major Project — Complete System Context Prompt

I have a major project called **Detective AI**. The project is an AI-powered crime investigation system built around CCTV footage and other evidence.

I originally built/vibe-coded much of the project, so I now need to understand exactly what the system does, how the components interact, and how to test it properly.

## 1. Main Goal

The goal of the project is:

> Given multiple pieces of evidence from a crime investigation — especially CCTV footage, access-control logs, and witness statements — the system should analyze the evidence, generate possible explanations/hypotheses, reason about movement between cameras, challenge its own conclusions, verify claims against evidence, and finally generate an investigation report.

The system should NOT simply say:

> "This person is the criminal."

Instead, it should behave more like an AI-assisted investigator:

```text
Evidence
   ↓
Evidence analysis
   ↓
Possible hypotheses
   ↓
Movement / trajectory reasoning
   ↓
Hypothesis evaluation
   ↓
Adversarial criticism
   ↓
Evidence verification
   ↓
Final investigation report
```

---

# 2. High-Level Architecture

The overall system can be understood as:

```text
                    USER / INVESTIGATOR
                           │
                           ▼
                     FRONTEND UI
                           │
                           │ HTTP
                           ▼
                       FASTAPI
                           │
                           ▼
                 INVESTIGATION PIPELINE
                           │
                           ▼
                       LANGGRAPH
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   INVESTIGATOR       TRAJECTORY          OTHER
          │                │
          └────────────────┼────────────────┘
                           ▼
                         CRITIC
                           │
                           ▼
                        VERIFIER
                           │
                           ▼
                        REPORTER
                           │
                           ▼
                    FINAL REPORT
```

Evidence enters the system through ingestion APIs and is stored in the database.

---

# 3. Types of Evidence

The system is designed to work with multiple evidence sources.

## CCTV

Example:

```text
Camera 1
10:31:12
Person detected

Camera 1
10:32:05
Person moves toward entrance

Camera 2
10:34:17
Person detected
```

## Access-control logs

Example:

```text
10:32:01
Door A opened

10:32:04
Access granted to employee ID 102
```

## Witness statements

Example:

```text
"Someone wearing a dark jacket
was seen near the entrance around 10:32."
```

These different evidence sources should eventually be combined.

---

# 4. Computer Vision Pipeline

The project contains a CV component.

The purpose is to process surveillance footage and extract useful observations.

Conceptually:

```text
CCTV Video
     ↓
Video frames
     ↓
Object / person detection
     ↓
Person tracking / identification
     ↓
Observations
     ↓
Evidence
```

The project uses YOLOv8n for object/person detection.

It also uses MobileNetV2-related embeddings/re-identification functionality to compare people appearing in different frames/cameras.

For example:

```text
Camera 1

Person image
     ↓
Feature extraction
     ↓
Embedding A
```

and:

```text
Camera 2

Person image
     ↓
Feature extraction
     ↓
Embedding B
```

Then the system can compare:

```text
similarity(A, B)
```

to estimate whether the observations could correspond to the same person.

Important:

A visual match should be treated as evidence/likelihood, NOT automatically as proof of identity.

---

# 5. Evidence vs Inference

This distinction is extremely important.

The system must distinguish between:

## Observed evidence

Example:

```text
Camera 1 detected a person at 10:31.
```

This is an observation.

## Inference

Example:

```text
The person probably moved from Camera 1
to Camera 2.
```

This is an inference.

The system should not represent an inference as if it were directly observed.

Conceptually:

```text
OBSERVED
   ↓
EVIDENCE

INFERRED
   ↓
HYPOTHESIS / UNCERTAINTY
```

---

# 6. Hypotheses

A hypothesis is a possible explanation of the evidence.

Example:

```text
Evidence:

E1 → Person A seen at Camera 1
E2 → Door opened at 10:32
E3 → Person A seen at Camera 2
```

Possible hypotheses:

```text
H1:
Person A entered through Door A
and moved toward Camera 2.

H2:
Person A left the area and
another person appeared at Camera 2.

H3:
The camera observations were
not the same person.
```

The system should maintain multiple hypotheses instead of immediately committing to one answer.

Hypotheses can be:

```text
created
evaluated
supported
contradicted
pruned
rejected
```

The system should preserve reasons for rejection when possible.

---

# 7. LangGraph

LangGraph is the orchestration framework used to implement the investigation workflow.

The most important concept is STATE.

Conceptually the investigation state might contain:

```python
state = {
    "case": ...,
    "evidence": [...],
    "hypotheses": [...],
    "current_hypothesis": ...,
    "trajectory": ...,
    "critic_findings": [...],
    "verification_results": [...],
    "confidence": ...,
    "final_report": ...
}
```

Different LangGraph nodes read and update this state.

Conceptually:

```text
State
  ↓
Investigator
  ↓
Updated State
  ↓
Trajectory Agent
  ↓
Updated State
  ↓
Critic
  ↓
Updated State
  ↓
Verifier
  ↓
Updated State
  ↓
Reporter
  ↓
Final Report
```

LangGraph therefore acts as the workflow/state-machine connecting the different AI components.

---

# 8. Agents

The project has six major conceptual agents.

## 8.1 Orchestrator

The orchestrator is the coordinator.

Its responsibility is to decide:

```text
What should happen next?
Which agent should run?
Which hypothesis should be examined?
Should the investigation continue?
```

Think of it as the investigation team lead.

---

## 8.2 Investigator

The investigator analyzes evidence and attempts to construct explanations.

Example:

```text
E1:
Person seen at Camera 1

E2:
Door opened

E3:
Person seen at Camera 2

        ↓

Investigator

        ↓

These observations may describe
a continuous movement path.
```

The investigator should use available evidence rather than inventing facts.

---

## 8.3 Trajectory Agent

The trajectory agent reasons about movement between observations.

Example:

```text
Camera A
   │
   │ person observed
   ▼
   ?
   ?
BLIND SPOT
   ?
   ?
   ▼
Camera B
   │
   │ person observed
```

The system may infer:

```text
Camera A → Camera B
```

but this is uncertain if no camera directly observed the movement.

The project uses camera topology / Markov-style reasoning for this component.

The important distinction is:

```text
Observed:
Camera A saw the person.

Inferred:
The person probably travelled
from A to B.
```

---

# 9. Critic Agent

The critic is an adversarial component.

Instead of agreeing with the leading hypothesis, it attempts to find weaknesses.

Example:

```text
Leading hypothesis:

Person A committed the shoplifting.
```

The critic asks:

```text
What evidence actually supports this?

Is there contradictory evidence?

Could another person explain the observations?

Are there gaps in the timeline?

Are we confusing inference with observation?
```

Example output:

```text
Issue 1:
Camera 2 does not clearly identify the person.

Issue 2:
Another person appears during the same time period.

Issue 3:
The inferred movement path is uncertain.
```

This prevents the system from blindly accepting the first plausible explanation.

---

# 10. Verifier Agent

The verifier checks whether claims made by the system are actually supported by evidence.

Example claim:

```text
"Person A entered the store at 10:32."
```

Verifier asks:

```text
Which evidence supports this claim?
```

For example:

```text
Claim
 ↓
Evidence E17
 ↓
Does E17 support the claim?
 ↓
YES / NO / PARTIALLY
```

The goal is to make claims traceable to evidence.

The final report should ideally be able to answer:

```text
Where did this claim come from?
```

---

# 11. Reporter Agent

The reporter takes the investigation results and generates a final report.

Input:

```text
Evidence
+
Hypotheses
+
Trajectory analysis
+
Critic findings
+
Verification results
```

Output:

```text
Investigation Report
```

The report should distinguish:

```text
Facts
Inferences
Uncertainty
Contradictions
Rejected hypotheses
Supporting evidence
```

It should not present uncertain conclusions as confirmed facts.

---

# 12. RAG / Embeddings

The project also contains text embedding / retrieval functionality.

The basic idea is:

```text
Evidence text
     ↓
Embedding model
     ↓
Vector
     ↓
Vector database
```

Later, when an agent needs relevant evidence:

```text
Agent query
     ↓
Query embedding
     ↓
Similarity search
     ↓
Relevant evidence
     ↓
Agent
```

This prevents the LLM from having to process every piece of evidence every time.

---

# 13. Database

The project supports persistent storage.

Conceptually:

```text
                 DATABASE
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
     Cases       Evidence      Embeddings
       │
       ├── CCTV observations
       ├── Access logs
       ├── Statements
       └── Hypotheses
```

PostgreSQL + pgvector can be used for production-style storage.

SQLite can be used for simpler development/testing environments.

---

# 14. MCP

The project contains an MCP interface.

The purpose is to expose tools that an AI agent can use.

Conceptually:

```text
LLM Agent
    │
    │ "Find evidence related to Camera 3"
    ▼
MCP Tool
    │
    ▼
Database
    │
    ▼
Relevant Evidence
    │
    ▼
LLM Agent
```

MCP therefore provides a standardized tool interface between the AI and external data/functions.

---

# 15. FastAPI

FastAPI is the backend API layer.

Conceptually:

```text
Frontend
   │
   │ HTTP request
   ▼
FastAPI
   │
   ▼
Application logic
   │
   ▼
Investigation pipeline
```

The backend exposes APIs related to:

```text
video ingestion
access logs
witness statements
investigations
reports
counterfactual analysis
```

---

# 16. Frontend

The frontend is the interface used by the investigator.

Conceptually:

```text
User
 ↓
Dashboard
 ↓
Upload evidence
 ↓
Start investigation
 ↓
Monitor results
 ↓
View hypotheses
 ↓
View evidence
 ↓
View final report
```

The frontend communicates with FastAPI rather than directly implementing the investigation logic.

---

# 17. Full End-to-End Flow

A complete investigation should conceptually work like this:

```text
                    CCTV VIDEO
                        │
                        ▼
                 COMPUTER VISION
                        │
                        ▼
                  OBSERVATIONS
                        │
                        │
ACCESS LOGS ────────────┤
                        │
WITNESS STATEMENTS ─────┤
                        ▼
                    EVIDENCE
                        │
                        ▼
                  DATABASE / RAG
                        │
                        ▼
                    LANGGRAPH
                        │
                        ▼
                  ORCHESTRATOR
                        │
                        ▼
                  INVESTIGATOR
                        │
                        ▼
                   HYPOTHESES
                        │
                        ▼
                   TRAJECTORY
                        │
                        ▼
                     CRITIC
                        │
                  ┌─────┴─────┐
                  │           │
             weaknesses    acceptable
                  │           │
                  └─────┬─────┘
                        ▼
                    VERIFIER
                        │
                        ▼
                    REPORTER
                        │
                        ▼
                FINAL REPORT
```

---

# 18. Example Test Case: Shoplifting

For testing the project, use a LONG surveillance video rather than a 5–10 second classification clip.

A suitable source is a long surveillance dataset such as UCF-Crime, specifically the Shoplifting category.

Example:

```text
Case:
Shoplifting Investigation

Video:
Shoplifting004_x264.mp4

Evidence:
CCTV Camera 1
```

The video may contain:

```text
Normal shopping
     ↓
Person enters
     ↓
Person moves around store
     ↓
Person interacts with products
     ↓
Possible concealment
     ↓
Person moves away
     ↓
Person exits
```

The system should not simply classify the entire video as:

```text
SHOPLIFTING = TRUE
```

Instead, it should attempt to identify relevant observations and build a timeline.

---

# 19. Full Shoplifting Investigation Example

Suppose CCTV produces:

```text
10:31:12
Person P1 enters store.

10:32:20
P1 approaches shelf.

10:33:05
P1 interacts with product.

10:33:21
P1 appears to conceal an object.

10:34:10
P1 moves toward exit.

10:34:42
P1 leaves store.
```

The system could create:

```text
Hypothesis H1:

P1 concealed merchandise
and left the store without paying.
```

But it should also consider:

```text
H2:

The object was returned later.

H3:

The apparent concealment was not actually
removal of merchandise.

H4:

The object was already possessed by P1.
```

Then the critic should attack H1.

The verifier should check every important claim.

Finally the reporter generates something like:

```text
Investigation Summary

Observed:
P1 approached the shelf at 10:32:20.

Observed:
P1 interacted with a product at 10:33:05.

Inferred:
P1 may have concealed an object at 10:33:21.

Observed:
P1 moved toward the exit.

Observed:
P1 exited at 10:34:42.

Uncertainty:
The available footage does not conclusively
establish whether the object was merchandise
or whether payment occurred elsewhere.

Conclusion:
The evidence is consistent with shoplifting,
but additional evidence is required for a
definitive conclusion.
```

This is the kind of reasoning the system should aim for.

---

# 20. Testing Strategy

Do NOT test everything simultaneously.

Test in layers.

## Test 1 — Video ingestion

```text
MP4
 ↓
API
 ↓
Video processing
 ↓
Stored correctly
```

## Test 2 — Computer vision

```text
MP4
 ↓
YOLO
 ↓
Person detection
```

## Test 3 — Tracking / Re-identification

```text
Person
 ↓
Embedding
 ↓
Compare observations
```

## Test 4 — Evidence

```text
CV output
 ↓
Evidence object
 ↓
Database
```

## Test 5 — Hypothesis generation

```text
Evidence
 ↓
Hypothesis
```

## Test 6 — Trajectory

```text
Camera A
 ↓
Blind region
 ↓
Camera B
```

## Test 7 — Critic

```text
Hypothesis
 ↓
Critic
 ↓
Contradictions / weaknesses
```

## Test 8 — Verification

```text
Claim
 ↓
Evidence IDs
 ↓
Verification
```

## Test 9 — Final report

```text
All investigation state
 ↓
Reporter
 ↓
Report
```

## Test 10 — Full integration

```text
Real CCTV
+
Synthetic logs
+
Synthetic witness statement
        ↓
Entire pipeline
        ↓
Final investigation report
```

---

# 21. Important Testing Principle

Real CCTV datasets generally provide the video, but they do NOT necessarily provide:

```text
access-control logs
witness statements
camera topology
ground-truth investigation reports
```

Therefore, for a complete end-to-end test, it is acceptable to combine:

```text
REAL CCTV FOOTAGE
        +
SYNTHETIC ACCESS LOGS
        +
SYNTHETIC WITNESS STATEMENT
        +
KNOWN CAMERA TOPOLOGY
```

The synthetic evidence must be clearly marked as synthetic.

Do not claim that synthetic logs/statements came from the real incident.

---

# 22. Current Objective

The immediate objective is NOT to build new features.

The objective is:

> **Understand and validate the existing Major Project.**

First determine:

1. How the application starts.
2. What the entry point is.
3. How a video enters the system.
4. How frames are processed.
5. How CV output becomes evidence.
6. How evidence reaches LangGraph.
7. What the LangGraph state contains.
8. What every agent receives.
9. What every agent returns.
10. How hypotheses are created and updated.
11. How trajectory reasoning works.
12. How the critic works.
13. How verification works.
14. How the report is generated.
15. Where the final result is stored/displayed.

For every component, explain:

```text
INPUT
 ↓
CODE / FUNCTION
 ↓
PROCESSING
 ↓
OUTPUT
 ↓
NEXT COMPONENT
```

Do not assume that a component works merely because the code exists.

Trace the actual execution path and identify:

```text
working
not working
partially implemented
placeholder
mock
synthetic
unused
```

The final goal is for me to be able to explain the entire project to a professor/interviewer and confidently demonstrate it using a real long shoplifting surveillance video.
