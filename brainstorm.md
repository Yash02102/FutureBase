Below is a practical deep dive into the major “agent types” you’ll actually encounter, with representative GitHub references for each. Real systems are usually hybrids, so treat these as building blocks rather than mutually exclusive boxes.


---

1. Simple reflex agents



What they are
Hard-coded condition → action rules using only the current observation (“if X, do Y”). No meaningful state, no planning. This maps cleanly to the classic “simple reflex” category.

When they win

Extremely reliable, low latency, easy to test and certify.

Best when the environment is fully observable and the policy is stable.

Where they break

Partial observability (you need memory).

Non-trivial tradeoffs (you need utility/objectives).

Typical implementation

Rule engine / decision table / finite-state machine.

GitHub references (closest “spirit” examples)

Workflow-orchestration frameworks are often used to implement reflex-like automation with explicit rules and dependencies (even if they’re not “AI agents”): apache/airflow.


---

2. Model-based reflex agents



What they are
Still rule-driven, but they maintain an internal state/world model inferred from history (e.g., last known values, belief state). Classic “model-based reflex.”

When they win

Partially observable environments where you can cheaply maintain state.

You want predictability but need “memory.”

Where they break

If the model is wrong, the agent confidently does the wrong thing.

Still limited without explicit goal/utility reasoning.

How to build

State estimator + rules. Often: event sourcing, belief store, or probabilistic filters.

GitHub references

Robotics navigation stacks are a canonical example of “model-based” behavior (localization + mapping + state + behaviors): ros-navigation/navigation2.


---

3. Goal-based agents



What they are
They choose actions to reach a goal state (planning/search). Classic “goal-based.”

When they win

Tasks where you can define success clearly (reach state G).

You can simulate or reason about action outcomes.

Common patterns

Classical planning (STRIPS-like), graph search, task planning, hierarchical planning.

Where they break

Multiple competing goals (needs utility).

Uncertainty/noisy outcomes (needs probabilistic planning or RL).

GitHub references (LLM-era “goal pursuit” orchestration)

langchain-ai/langgraph (stateful, long-running agent workflows; graphs are a clean way to encode goal pursuit and control flow).

microsoft/semantic-kernel (agent orchestration and planners/function-calling patterns).

Strong opinion: if someone says “agent” in 2026, 80% of the time it’s a goal-based loop with tools, not a new intelligence class.


---

4. Utility-based agents



What they are
They maximize expected utility, not just “goal reached / not reached.” This is how you encode tradeoffs (cost, risk, latency, user preference). Classic “utility-based.”

When they win

Ranking, selection, scheduling, bidding, routing, resource allocation.

Any scenario with explicit tradeoffs and uncertainty.

Where they break

Garbage utility function = garbage behavior.

Over-optimization can produce “clever” but undesirable strategies unless constrained.

How to build (practical)

Scoring model + constraints (hard rules) + calibration.

Often paired with bandits or RL for online adaptation.

GitHub references (foundational tooling)

RL libraries can be used to learn policies that approximate utility maximization: DLR-RM/stable-baselines3, ray-project/ray (RLlib).


---

5. Learning agents (RL / imitation / online adaptation)



What they are
Agents that improve their policy from data/feedback. In practice this spans:

Reinforcement learning (reward-driven)

Imitation learning (demonstrations)

Bandits/online learning (incremental improvement)

This aligns with “learning agents” in the classic taxonomy.

When they win

The environment is too complex to hand-code.

You can define reward/feedback and run many trials (simulators help).

Where they break

Reward hacking, brittle policies under distribution shift, high ops burden.

Training is often more expensive than people expect.

GitHub references

RL algorithms: DLR-RM/stable-baselines3.

Standard RL environments: Farama-Foundation/Gymnasium.

Multi-agent RL environments: Farama-Foundation/PettingZoo.

Scalable RL: RLlib in ray-project/ray.


---

6. BDI agents (Belief–Desire–Intention)



What they are
A symbolic agent architecture: beliefs (world model), desires (objectives), intentions (committed plans). This is a mainstream approach in multi-agent systems and agent-oriented programming.

When they win

You need interpretability: “why did the agent do that?”

You want explicit plan libraries + deterministic-ish execution.

Where they break

If you need continuous control / high-dimensional perception, pure BDI struggles (often hybridized with ML/RL).

GitHub references

jason-lang/jason (AgentSpeak/BDI interpreter; great for learning the model).

actoron/jadex (BDI reasoning engine for Java; industrial MAS lineage).

Interesting hybrid: BDI + RL integration examples exist (e.g., Jason + RL).


---

7. Tool-using LLM agents (single-agent)



What they are
An LLM wrapped in a control loop with tool/function calls, state, and retry/validation. Most modern “agents” are this.

Core risks (the ones that matter)

Tool hallucination without strict schemas/validation.

Prompt injection and untrusted tool outputs.

Non-determinism breaks tests unless you design for it.

GitHub references

langchain-ai/langchain (agent abstractions + tool integrations).

langchain-ai/langgraph (more explicit state machines/graphs; generally more production-friendly).

microsoft/semantic-kernel (enterprise-oriented orchestration, planners).

run-llama/llama_index and example multi-agent workflows over data (RAG-centric agent patterns).


---

8. Multi-agent systems (role teams, supervisors, debates)



What they are
Multiple agents with distinct roles (planner/executor/critic; researcher/writer/reviewer), coordinating via messages or a supervisor. Good for decomposition, parallelism, and separating concerns.

When they win

Complex tasks that benefit from specialization and cross-checking.

You can tolerate extra latency/cost for higher quality.

Where they break

Coordination overhead; can become “agents talking to themselves” without grounding.

Debugging is harder unless you impose structure.

GitHub references

microsoft/autogen (multi-agent framework; note their docs point newcomers to Microsoft Agent Framework).

microsoft/agent-framework (orchestration/deployment focus).

crewAIInc/crewAI and crewAIInc/crewAI-examples (role-based multi-agent teams).

openai/swarm (lightweight educational multi-agent orchestration).


---

9. Web agents and browser-automation agents



What they are
Agents that navigate websites, fill forms, extract data. Under the hood: browser automation + planning + strong guardrails.

When they win

Legacy systems with no APIs.

Competitive intel / repetitive ops tasks.

Where they break

Websites change constantly; selectors break.

Security risk: prompt injection via web content unless you sandbox and sanitize.

GitHub references

browser-use/browser-use (LLM-oriented web interaction).

Traditional automation foundation: SeleniumHQ/selenium.


---

10. Code agents (software engineering agents)



What they are
Agents that modify repos, run tests, open PRs, etc. The main differentiator is tight tool integration (git, CI, test runners) and verification loops.

Where they succeed or fail

Succeed when tests are strong and tasks are well-scoped.

Fail when specs are ambiguous and the repo lacks guardrails (lint, tests, type checks).

GitHub references

OpenHands/OpenHands (platform/SDK for coding agents).


---

If you want, I can map your system to this taxonomy in a precise way

I was researching a bit
Suggest for accuracy


# Middleware Ideas
Common custom middleware ideas for this project:

Token budget limiter / auto‑summarize when messages grow too large.
Tool allowlist / rate limiting / per‑tool quotas.
PII scrubbing on tool outputs before they re‑enter the model context.
Structured logging / metrics for model and tool latency.
Policy injection via dynamic_prompt to adjust system prompt per request.
Response normalization (e.g., enforce citations when RAG is used).

Input / prompt safety

Max input length + hard truncation
Prompt‑injection detection (block or strip “ignore system”, “exfiltrate secrets”)
PII detection/redaction before tools or logs
Tool safety

Tool allowlist by intent (e.g., no checkout unless intent is purchase)
Tool argument validation (schema + value bounds)
Rate limiting / retries with backoff
“Dangerous tool” approvals (checkout, refunds, address changes)
Commerce / business logic

Price ceiling guardrail (e.g., reject items above user budget)
Inventory must be available before cart/checkout
Require user confirmation on cart + payment + address
Order total validation against promotions
RAG reliability

Minimum relevance threshold (return “need more info” if low score)
Require citations when RAG context is used
Block hallucinated SKUs (must exist in catalog tool output)
Data access

Filesystem root enforcement (limit to FILESYSTEM_ROOT)
MCP server allowlist (only known hosts/tools)
Prevent tool calls that include secrets or paths outside allowed root
Operational

Max tokens per run; summarize when exceeded
Cache‑hit guardrail (avoid repeated expensive calls)
Observability: log all tool calls and decisions
Where to wire them

Middleware (wrap_tool_call, wrap_model_call) for tool gating, retries, and PII scrub
Graph nodes for input/output checks + RAG confidence checks
MCP server for schema and policy enforcement


# Muti agent

## Subagents - Supervisor pattern
The supervisor pattern is a multi-agent architecture where a central supervisor agent coordinates specialized worker agents. This approach excels when tasks require different types of expertise. Rather than building one agent that manages tool selection across domains, you create focused specialists coordinated by a supervisor who understands the overall workflow.

## Handoff - State machine pattern

The state machine pattern describes workflows where an agent’s behavior changes as it moves through different states of a task. This tutorial shows how to implement a state machine by using tool calls to dynamically change a single agent’s configuration—updating its available tools and instructions based on the current state. The state can be determined from multiple sources: the agent’s past actions (tool calls), external state (such as API call results), or even initial user input (for example, by running a classifier to determine user intent).

## Router

The router pattern is a multi-agent architecture where a routing step classifies input and directs it to specialized agents, with results synthesized into a combined response. This pattern excels when your organization’s knowledge lives across distinct verticals—separate knowledge domains that each require their own agent with specialized tools and prompts.

## SKILLS

Progressive disclosure - a context management technique where the agent loads information on-demand rather than upfront - to implement skills (specialized prompt-based instructions). The agent loads skills via tool calls, rather than dynamically changing the system prompt, discovering and loading only the skills it needs for each task.

--------------------------------------------------------


# Agents core conepcts

## Context Magement

- Models context
- Tools context
- lifecyle context

Large context
Manage context as file system - this means that results can put to files rather than dumpping to LLM and tools call result will be replaced by this file location and agent can access this like a tool getting data it needs

## Plugable memory system

State Backend - Epihermal memory
File System - On Disk
Store Backend - long term memory
Composite Backend - Route different path to different paths

## Task Delegation - Subagents

context isolation
Parallel execution
Specializtion
Token efficiency

## Coversational memory summurization

enables long conversation hitting limits
context recent mesaages while compressing ancient history

## Dangling tool calls

fixes message history so AI does not gets confused what happened after AI Message
Perevents confusion
Garcefully handle failures
Maintains convesration coherence

## Todo list tracking

Track mutiple tasks status
Helpful for long horizon tasks
Organize complex muti step tasks

## Human in the loop

safety gates for desctuctive oprations
user verification before expensive API calls


## Caching

reduce token consumption
reduce context flodding
reduce load on tools calls (MCP/API/etc..)
reduce rate limit hitting
high respone on repeated questions

## MCP

## RAG

------------------------------

# Observability

## Runs (spans) 

single unit of work done by LLM apploication

## Traces

Collection of Runs that an complete operation

## Theards

Sequence of traces - muti turn conversation

## Projects

Collection of traces

## Feedback

allows to tag or score each run 

## Tags

## Metadata


--------------------------------------

# Evaluation

## Offline
- Create dataset
- Define evalutors
    - human review
    - code evalutor
    - LLM as a judge
    - pairwise 
- Run Experiments
- analyze

## Online

- Deploy
- online evalutors
- Monitor in real time
- Establish a feedback loop