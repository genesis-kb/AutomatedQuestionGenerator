# AutomatedQuestionGenerator (`aqg`)

`aqg` reads Bitcoin conference-talk transcripts from the **transcription_engine** Postgres database and turns
them into coding problems for the **hello-bitcoin** online judge. Each problem comes with a statement, starter
code, a hidden wrapper, reference solutions and test cases. Before it is written out, every problem is checked
inside hello-bitcoin's own judge image. Neither sibling repo is ever modified: the database is read through a
connection that enforces read-only access, the judge image is built from hello-bitcoin's existing Dockerfile,
and every run compares `git status` of both repos before and after. For the full design and data flow, see
[ARCHITECTURE.md](ARCHITECTURE.md).

There are two commands:

```bash
aqg generate   # transcripts -> validated problems (local JSON in out/)
aqg test       # run the test suite (builds the judge image first if Docker is up)
```

Everything else comes from `config.yaml` (non-secret settings) and `.env` (secrets).

## Prerequisites

- **conda env `sob`** with Python 3.11 (`conda activate sob`).
- **Docker**, running. The judge runs inside the `hello-bitcoin-runner` image.
- **An LLM.** `config.yaml` currently uses Kimi K3 (Moonshot AI) on Amazon Bedrock (`global.moonshotai.kimi-k3`,
  in `ap-south-2`) for every stage, as a trial, at effort `high` for the steps that decide correctness and `medium`
  for triage and the statement prose. Claude Opus 4.6 (`global.anthropic.claude-opus-4-6-v1`, `bedrock_endpoint:
  runtime`) is kept commented out right below it; switch its model to `global.anthropic.claude-opus-5-5` once the
  AWS account has Opus 5.5 access. No key goes in `.env`: credentials come from
  the EC2 instance's IAM role (or `AWS_PROFILE` locally), and `AWS_REGION` picks the region. Setup:
  [Amazon Bedrock (Claude)](#amazon-bedrock-claude). To switch to another provider (Groq is kept commented out
  in `config.yaml`), including free local Ollama, see [Appendix: LLM providers](#appendix-llm-providers). Small local models such as `gemma3:4b` are fine for
  triage but usually fail at writing specs and reference code.
- **`../hello-bitcoin`** checked out next to this repo. The runner image is built from
  `../hello-bitcoin/docker`.
- **`../transcription_engine`** checked out next to this repo, with its Postgres database.

## Running locally

**1. Start the transcription_engine database** (Postgres on port 5434):

```bash
cd ../transcription_engine
docker compose up -d postgres
python transcriber.py db init
```

**2. Transcribe a talk with transcription_engine.** transcription_engine needs its own configuration first:
- `transcription_engine/.env` must contain
  `DATABASE_URL=postgresql://bitcoin:bitcoin@127.0.0.1:5434/transcription_engine`.
- `transcription_engine/config.ini` must set `llm_provider = gemma`, plus
  `llm_correction_model = gemma3:4b` and `llm_summary_model = gemma3:4b`.

Then run the transcription. For example, with Whisper and Gemma:

```bash
python transcriber.py transcribe "<youtube url>" \
  --asr-provider whisper -m small.en --correct --summarize --llm-provider gemma \
  --json --markdown --nocheck --loc misc --username local
```

The transcript is stored in the database. The JSON and Markdown files go to `transcription_engine/local_models/`
and `metadata/`.

**3. Install aqg:**

```bash
cd ../AutomatedQuestionGenerator
conda activate sob
pip install -e ".[dev,bedrock,gemini]"
cp .env.example .env        # the defaults work for local Ollama + local Postgres
```

`TAVILY_API_KEY` in `.env` is optional. When it is set, S4 grounds each problem with web search over the
allowlisted domains. When it is not set, grounding falls back to `transcript_only`.

**4. Run the tests:**

```bash
aqg test                    # extra args go to pytest, e.g. `aqg test tests/unit` or `aqg test -k deepseek`
```

If Docker is running, `aqg test` first builds `hello-bitcoin-runner` (from `sandbox.build_dir`), so the judge
integration tests actually run. Tests that need Postgres or Ollama are skipped when those services are down.

**5. Generate problems:**

```bash
aqg generate                                   # the most codeable talk not used yet
aqg generate --limit 3                         # the 3 most codeable unused talks
aqg generate --content-item <uuid>             # one specific talk (transcription_engine content_items.id)
aqg generate --from-file talk.md               # a transcript file instead of the DB (.md, .txt, whisper JSON)
aqg generate --langs python3,javascript,rust   # or --langs all; default: `languages` in config.yaml
aqg generate --resume <run_id>                 # continue a run that failed part-way (see below)
```

**Resuming.** Every step of a run is saved as JSON under `runs/<run_id>/` (concepts, then per concept
`s3_spec.json`, `s4_grounding.json`, `s5_statement.json`, ...). `aqg generate --resume <run_id>` runs the same talk
again in the same folder. It reuses every saved step and calls the LLM only for steps that are missing or had
failed. To redo a step on purpose (for example after switching it to a better model), delete or rename that file
and resume.

If the judge image is missing, the first `generate` builds it. The terminal shows only a summary: accepted and
rejected problems with their paths, LLM call and token counts, and where the logs are.

**Where the outputs go:**

| Path | Contents |
|---|---|
| `out/problems/` | accepted problems, one JSON file each |
| `out/rejected/` | rejected concepts and problems, with the reason |
| `out/manifests/` | one manifest per run (parameters, models, prompt hashes, results) |
| `runs/<run_id>/` | per-run artifacts for every stage (see [Pipeline](#pipeline)) and `run.log` |
| `logs/` | `aqg.log` and `errors.log` (see [Logging](#logging)) |

**Languages.** `python3` is always generated, because its reference computes the expected outputs. JavaScript
and Rust are added only when you request them. When they are included, their references must reproduce the
Python outputs exactly on every test.

## Configuration

`config.yaml` holds the main settings:

| Key | Default | Meaning |
|---|---|---|
| `languages` | `[python3]` | judge languages to generate (`python3`, `javascript`, `rust`) |
| `llm.<stage>` | `bedrock` / `global.moonshotai.kimi-k3` (`ap-south-2`) | provider, model, temperature, optional `reasoning_effort` (groq / openai_compat: low/medium/high; anthropic / bedrock Claude: low/medium/high/xhigh/max; bedrock Kimi K3: none/low/medium/high/xhigh/max) and optional `aws_region` (bedrock) per stage; stages not listed use `llm.default` |
| `llm.request_timeout_s` | `1800` | per-request LLM timeout |
| `pipeline.max_problems_per_talk` | `2` | the maximum number of problems produced from one talk |
| `pipeline.min_codeability` | `3` | the minimum triage score for a talk to be used |
| `pipeline.max_repair_loops` | `3` | how many times failed validation is routed back for fixes |
| `pipeline.min_hidden_tests` | `10` | the minimum number of hidden tests per problem |
| `pipeline.time_limit_s` / `memory_limit_mb` | `2.0` / `256` | limits written into each problem |
| `sandbox.image` / `build_dir` | `hello-bitcoin-runner` / `../hello-bitcoin/docker` | the judge image and where it is built from |
| `sandbox.workers` | `4` | parallel sandbox executions |
| `search.allow_domains` | bitcoin docs sites | domains Tavily results may come from |
| `paths.protected_repos` | the two sibling repos | repos whose `git status` must not change |

`.env` holds the secrets:

| Variable | Needed when |
|---|---|
| `TE_DATABASE_URL` | reading from the DB (default `postgresql+psycopg://bitcoin:bitcoin@127.0.0.1:5434/transcription_engine`) |
| `TAVILY_API_KEY` | optional; enables web grounding |
| `OLLAMA_HOST` | provider `ollama` (default `http://127.0.0.1:11434`) |
| `ANTHROPIC_API_KEY` | provider `anthropic` |
| `GOOGLE_API_KEY` | provider `gemini` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | provider `openai_compat` |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | provider `deepseek` |
| `GROQ_API_KEY`, `GROQ_BASE_URL` | provider `groq` |
| `AWS_REGION` | provider `bedrock` (optional; a stage's `aws_region` wins, default `us-east-1`). No AWS key is needed on EC2 |
| `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | provider `bedrock` on a local machine only (read by boto3) |

## Logging

All logging goes to files. The terminal shows only command results and the paths to the logs.

| File | Contents |
|---|---|
| `runs/<run_id>/run.log` | **start here.** INFO and above for one run, one readable line per event, long values shortened |
| `logs/aqg.log` | everything (DEBUG+) across runs, in full: prompts and raw LLM output are printed indented under their line. Rotating 10 MB × 5 |
| `logs/errors.log` | ERROR and CRITICAL only, with tracebacks |

Lines look like this:

```
2026-10-16 21:23:39 INFO  [S3 spec] ▶ started | concept=CompactSize integer encoding
2026-10-16 21:23:41 INFO  [S3 spec] llm s3_spec ok | model=bedrock:global.anthropic.claude-opus-4-6-v1 latency=1.9s tokens_in_out=1220/154
2026-10-16 21:23:41 WARNING [S3 spec] llm rate limited; waiting as instructed | wait_s=12.8 wait=1
2026-10-16 21:23:55 INFO  [S3 spec] ✓ finished | took=15.2s concept=CompactSize integer encoding
```

API keys, the DB URL and any `user:password@` in URLs are masked.

How the levels are used:
- **DEBUG**: prompts and outputs, plus every sandbox execution.
- **INFO**: stage start and finish with durations, plus LLM call summaries.
- **WARNING**: fallbacks, dropped concepts and tests, repair loops, `transcript_only` grounding.
- **ERROR**: stage and concept failures.
- **CRITICAL**: a write attempt on the source DB, Docker unavailable, or a protected repo changed.

## Pipeline

| Stage | What | Output (in `runs/<run_id>/<talk>/`) |
|---|---|---|
| S1 | Read-only fetch, cleanup, structure-aware chunking with character offsets | `s1_document.json`, `s1_chunks.json` |
| S2 | Keyword + LLM triage; per-chunk concept extraction with a verbatim-excerpt check (rapidfuzz) | `s2_concepts.json` |
| S3 | Language-neutral `CodingSpec`: one pure function, types from a closed set | `c*/s3_spec.json` |
| S4 | Tavily web search over allowlisted domains (cached), when `TAVILY_API_KEY` is set; LLM fact check; literal test vectors. Without a key: `transcript_only` | `c*/s4_grounding.json` |
| S5 | LLM writes the statement, the reference per language and the test inputs. Wrappers and starters are generated by code. Expected outputs come from running the Python reference | `c*/s5_*.json` |
| S6 | Validation in hello-bitcoin-runner: references AC twice, cross-language agreement, starters compile but fail, resource headroom, test distribution, vectors, examples, static std-only rules | `c*/s6_report.json` |
| S7 | Atomic export to `out/problems/`, rejects to `out/rejected/`, manifest to `out/manifests/` | |

Failed S6 checks are routed back to the step that can fix them (statement, tests, or one language's
reference), up to `max_repair_loops` times.

## Running on AWS

The same two commands apply (`aqg test`, then `aqg generate`). The step-by-step EC2 setup (IAM role, Bedrock
model access, Python, Docker, Postgres) is in [Amazon Bedrock (Claude)](#amazon-bedrock-claude) and its
[Running on EC2](#running-on-ec2) subsection. In short:
- The LLM is Claude on Bedrock, authenticated by the instance's IAM role, so `.env` holds no LLM key.
- Docker must be available on the host, because the judge image is built and run there. The host also needs
  `../hello-bitcoin` next to this directory for the Dockerfile.
- `TE_DATABASE_URL` points at the transcription_engine Postgres: on the same instance it is the local one
  (`postgresql+psycopg://bitcoin:bitcoin@127.0.0.1:5434/transcription_engine`), with RDS it is
  `postgresql+psycopg://<user>:<password>@<rds-host>:5432/<db>`.

## Appendix: LLM providers

Set each stage in `config.yaml` under `llm:`. The stages are `s2_triage`, `s2_extract`, `s3_spec`,
`s4_reconcile`, `s5_statement`, `s5_reference` and `s5_tests`, and stages that are not listed use `default`.
Only the keys of the providers you reference are needed in `.env`.

**Mixing providers.** Triage and extraction are easy, so a cheap or local model is enough for `s2_triage` and
`s2_extract`. Specs and reference code are the hard part, and a strong model on `s3_spec` and `s5_reference`
sharply reduces repair loops:

```yaml
llm:
  default:      {provider: ollama,    model: "gemma3:4b",       temperature: 0.2}
  s2_triage:    {provider: ollama,    model: "gemma3:4b",       temperature: 0.0}
  s2_extract:   {provider: ollama,    model: "gemma3:4b",       temperature: 0.1}
  s3_spec:      {provider: anthropic, model: "claude-opus-5-5", temperature: 0.1}
  s5_reference: {provider: anthropic, model: "claude-opus-5-5", temperature: 0.0}
```

### ollama

- `.env`: `OLLAMA_HOST=http://127.0.0.1:11434` (this is the default).
- Recommended model: `gemma3:4b` (run `ollama pull gemma3:4b`). Any pulled model works.
- Notes: runs locally with no key. Output is constrained to the JSON schema, and the context window is 16k
  tokens. Small models work well for triage but cause more repair loops on S3/S5.

```yaml
llm:
  default: {provider: ollama, model: "gemma3:4b", temperature: 0.2}
```

### anthropic

- `.env`: `ANTHROPIC_API_KEY=...`. Install with the `anthropic` extra (`pip install -e ".[anthropic]"`).
- Recommended models: `claude-opus-5-5` (default, strongest), `claude-sonnet-5-5`, and `claude-haiku-5-5`
  (cheap, good for S2).
- Notes: uses structured outputs. The newest models ignore `temperature` because the API rejects it, so
  `aqg` does not send it to them.

```yaml
llm:
  default:   {provider: anthropic, model: "claude-opus-5-5", temperature: 0.2}
  s2_triage: {provider: anthropic, model: "claude-haiku-5-5", temperature: 0.0}
```

### Amazon Bedrock (Claude)

Provider `bedrock` (the shipped `config.yaml`). It reuses the `anthropic` client with the SDK's
`AnthropicBedrockMantle` class, which calls Claude in Amazon Bedrock at
`https://bedrock-mantle.<region>.api.aws/anthropic/v1/messages` with the normal Messages API and SigV4-signed
requests. Credentials come from the default AWS chain (EC2 instance role, `AWS_PROFILE`, or `AWS_*` keys), so
nothing goes in `.env` except, optionally, `AWS_REGION`.

- **Non-Claude models (Kimi K3, the shipped config).** When a `bedrock` stage's model ID has no `claude-` in it,
  aqg calls Bedrock's OpenAI-compatible Chat Completions instead
  (`https://bedrock-runtime.<region>.amazonaws.com/openai/v1/chat/completions`), SigV4-signed with the same
  default AWS credentials (no Bedrock API key). `bedrock_endpoint` is ignored for these. Kimi K3 IDs:
  `global.moonshotai.kimi-k3` (cheapest), `in.moonshotai.kimi-k3` (India only), `us.moonshotai.kimi-k3`; it
  reasons at `max` effort unless told otherwise, so every stage sets `reasoning_effort`. The IAM policy below
  must list `foundation-model/moonshotai.*`, and AWS Marketplace may ask for a one-time subscription on first use.
  Quick check from the instance:
  `aws bedrock-runtime converse --region ap-south-2 --model-id global.moonshotai.kimi-k3 --messages '[{"role":"user","content":[{"text":"Say hi"}]}]'`.
- Claude config (commented out in `config.yaml`): `global.anthropic.claude-opus-4-6-v1` on the runtime endpoint in `ap-south-2` (the account has no
  Opus 5.5 access yet), same efforts as below. Which models an account can call in a region:
  `AnthropicBedrock(aws_region=...)` plus a tiny `messages.create` per ID from `list_inference_profiles`; a 403
  "not available for this account" means no model access.
- Models on the mantle endpoint (IDs carry the `anthropic.` prefix): `anthropic.claude-opus-5-5` for every stage, effort `high` for
  `s2_extract`, `s3_spec`, `s4_reconcile`, `s5_reference` and `s5_tests`, `medium` for `default`, `s2_triage` and
  `s5_statement`. `anthropic.claude-fable-5-1` (most capable, about 2.5x the price) is the upgrade for the hard
  steps; `anthropic.claude-sonnet-5-5` is the cheaper option.
- Region: a stage's `aws_region` in `config.yaml`, else `AWS_REGION` / `AWS_DEFAULT_REGION`, else `us-east-1`.
  The Bedrock endpoint offers global routing (no premium) and regional routing (10% premium, for data residency).
  Use `ap-south-1` (Mumbai) from India. Anthropic's docs list `ap-south-2` (Hyderabad), but as of 2026-10-18
  `bedrock-mantle.ap-south-2.api.aws` does not resolve in DNS (`Name or service not known`). An EC2 instance in
  Hyderabad can call the Mumbai endpoint; only `AWS_REGION` changes. To stay in Hyderabad instead, add
  `bedrock_endpoint: runtime` and `aws_region: ap-south-2` to each stage: aqg then uses the SDK's `AnthropicBedrock`
  client against `bedrock-runtime.ap-south-2.amazonaws.com` (InvokeModel, same request body, IAM action
  `bedrock:InvokeModel`). Model IDs on that endpoint are usually inference profiles such as
  `global.anthropic.claude-opus-5-5`; list them with
  `python -c "import boto3; print([p['inferenceProfileId'] for p in boto3.client('bedrock', region_name='ap-south-2').list_inference_profiles()['inferenceProfileSummaries']])"`. Mumbai offers the **Global** endpoint only:
  Bedrock may serve a request from any region (no premium). Check a region with
  `getent hosts bedrock-mantle.<region>.api.aws`.
- Notes: `aqg` sends each stage's `reasoning_effort` as Claude's `effort` (default `medium`) and never
  `temperature` (these models reject sampling parameters). At `high` effort a call can think for several minutes,
  hence `request_timeout_s: 1800` and `max_tokens` 64000.
  Anthropic's Bedrock page lists structured outputs as not supported on this endpoint, so `aqg` tries
  `messages.parse` once and, if Bedrock rejects it, uses `messages.create` with the JSON schema in the system
  prompt for the rest of the run (the repair loop validates the answer as usual). Throttling (HTTP 429 /
  `ThrottlingException`, `ServiceUnavailableException`) is retried; missing credentials and `AccessDenied` stop
  the run with a message saying what to fix.

```yaml
llm:
  default:      {provider: bedrock, model: "anthropic.claude-opus-5-5", temperature: 0.2, reasoning_effort: medium}
  s3_spec:      {provider: bedrock, model: "anthropic.claude-opus-5-5", temperature: 0.2, reasoning_effort: high}
  s5_reference: {provider: bedrock, model: "anthropic.claude-opus-5-5", temperature: 0.2, reasoning_effort: high, aws_region: us-west-2}
```

**1. Enable model access** (once per AWS account and region). In the AWS console open
**Amazon Bedrock → Model access** in the region you will use (for example `ap-south-1`, Mumbai), choose
**Modify model access**, and enable Claude Opus 5.5 (and Claude Fable 5.1 if you plan to use it).
The first time you enable an Anthropic model, Bedrock asks you to fill in the one-time Anthropic use-case form;
access is usually granted within minutes. Access criteria are set per model, so check the status column says
*Access granted* for each one.

**2. Create an IAM policy** (IAM → Policies → Create policy → JSON), named for example `aqg-bedrock-invoke`.
Claude in Amazon Bedrock authorizes inference with `bedrock-mantle:CreateInference`. The second statement covers
the classic `bedrock-runtime` InvokeModel path (and cross-region inference profiles) so the same role also works
if you ever switch to `us.`/`global.` inference-profile IDs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ClaudeInBedrock",
      "Effect": "Allow",
      "Action": ["bedrock-mantle:CreateInference"],
      "Resource": "*"
    },
    {
      "Sid": "BedrockRuntimeInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:GetInferenceProfile"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.*",
        "arn:aws:bedrock:*::foundation-model/moonshotai.*",
        "arn:aws:bedrock:*:<ACCOUNT_ID>:inference-profile/*"
      ]
    },
    {
      "Sid": "BedrockList",
      "Effect": "Allow",
      "Action": ["bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
      "Resource": "*"
    },
    {
      "Sid": "MarketplaceSubscribeOnFirstUse",
      "Effect": "Allow",
      "Action": ["aws-marketplace:ViewSubscriptions", "aws-marketplace:Subscribe"],
      "Resource": "*"
    }
  ]
}
```

`Resource: "*"` on the first statement is deliberate: narrow it to your model ARNs once you have confirmed their
exact form in the Bedrock console. A cross-region inference profile (`us.` / `global.` prefix) routes requests to
several regions, so the foundation-model ARNs must be allowed in **every** region the profile routes to; the
`arn:aws:bedrock:*::foundation-model/...` wildcard above does that. If you list regions explicitly instead, list
all of them. The `*` in the region part of each ARN is deliberate too: with the Global endpoint (the only one in
`ap-south-1`) Bedrock may serve a request from another region, so don't pin the ARNs to one region. The list
actions don't support resource ARNs, hence their own `"Resource": "*"` statement.

The last statement matters once per model: the first call to a Claude model subscribes the AWS account to it in
AWS Marketplace, and that call fails with 403 "not authorized to perform the required AWS Marketplace actions" if
the caller can't subscribe. Alternatively, send one message to each model in the Bedrock console playground as an
admin, which subscribes the account; the role then doesn't need the statement. After a fix, wait about 5 minutes.

You can attach this policy to your existing IAM user instead of a role (then run `aws configure` on the instance
with that user's access keys). It works, but the keys then sit on the server and carry all of that user's
permissions; the role in steps 3 and 4 gives the instance short-lived credentials that only allow Bedrock.

**3. Create an IAM role for EC2** (IAM → Roles → Create role → Trusted entity *AWS service*, use case **EC2**),
attach `aqg-bedrock-invoke`, and name it for example `aqg-ec2-bedrock`. The console also creates an instance
profile with the same name.

**4. Attach the role to the instance.** Console: EC2 → Instances → select the instance → **Actions → Security →
Modify IAM role** → choose `aqg-ec2-bedrock` → **Update IAM role**. Or from the CLI:

```bash
aws ec2 associate-iam-instance-profile \
  --instance-id i-0123456789abcdef0 \
  --iam-instance-profile Name=aqg-ec2-bedrock
```

**5. Set the region and install** (on the instance, in `AutomatedQuestionGenerator/`):

```bash
echo 'AWS_REGION=ap-south-1' >> .env      # or export AWS_REGION=ap-south-1
pip install -e '.[bedrock]'              # anthropic[bedrock] brings boto3/botocore
```

**6. Verify.** `aws sts get-caller-identity` should show `assumed-role/aqg-ec2-bedrock/...`. Then:

```bash
aws bedrock list-foundation-models --region ap-south-1 --by-provider anthropic --query "modelSummaries[*].modelId"
aws bedrock list-inference-profiles --region ap-south-1 --query "inferenceProfileSummaries[*].inferenceProfileId"
python -c "
import anthropic
c = anthropic.AnthropicBedrockMantle(aws_region='ap-south-1')
for model in ('anthropic.claude-opus-5-5',):
    m = c.messages.create(model=model, max_tokens=2000, messages=[{'role': 'user', 'content': 'Say ok'}])
    print(model, ''.join(b.text for b in m.content if b.type == 'text') or m.stop_reason)
"
```

A `PermissionDeniedError` from the smoke test means the policy or model access is missing; "could not resolve
credentials" means the role is not attached (or, inside Docker, see the IMDS note below).

**7. Run:**

```bash
aqg test -rs
aqg generate --content-item <uuid>
```

**Local machine instead of EC2.** Use your own AWS identity: run `aws configure` (writes `~/.aws/credentials`)
or `aws sso login --profile <name>`, then `export AWS_PROFILE=<name>` and `export AWS_REGION=ap-south-1` (both can
also go in `.env`). That identity needs the same policy as step 2. Plain `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` in `.env` also work; `aqg` masks them in its logs. Nothing else changes.

#### Running on EC2

Clone the three repos into the same parent folder: the judge image is built from `../hello-bitcoin/docker`, and
every run compares the siblings' `git status`, so they must be real git checkouts.

```bash
mkdir -p ~/code/AQG && cd ~/code/AQG
git clone https://github.com/Rasesh2005/AutomatedQuestionGenerator.git
git clone https://github.com/genesis-kb/hello-bitcoin.git
git clone https://github.com/genesis-kb/transcription_engine.git   # only if Postgres runs on this machine
cp AutomatedQuestionGenerator/.env.example AutomatedQuestionGenerator/.env   # .env is never committed
```

`runs/`, `logs/`, `out/` and the ledger are git-ignored and created on the first run. On
Amazon Linux 2023 (on Ubuntu use `apt-get install -y python3.11 python3.11-venv docker.io docker-compose-v2`):

```bash
# Python 3.11 and a virtualenv
sudo dnf install -y python3.11 git
cd ~/QuestionPipeline/AutomatedQuestionGenerator
python3.11 -m venv .venv && . .venv/bin/activate && pip install -e '.[bedrock]'

# Docker, with your user in the docker group (log out and back in afterwards)
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
# docker compose plugin (Amazon Linux does not ship it)
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m) \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Postgres from transcription_engine's compose file (user/pass bitcoin, db transcription_engine, port 5434)
cd ~/QuestionPipeline/transcription_engine && docker compose up -d postgres

# aqg settings
cd ~/QuestionPipeline/AutomatedQuestionGenerator
cp .env.example .env
#   edit .env: TE_DATABASE_URL=postgresql+psycopg://bitcoin:bitcoin@127.0.0.1:5434/transcription_engine
#              AWS_REGION=ap-south-1
aqg test -rs        # builds hello-bitcoin-runner on first use
```

`aqg` itself runs on the host and talks to the instance metadata service (IMDS) directly, so the default IMDSv2
hop limit of 1 is fine. Only if you run `aqg` **inside** a container does the hop limit matter: raise it with
`aws ec2 modify-instance-metadata-options --instance-id <id> --http-put-response-hop-limit 2 --http-endpoint enabled`.
The judge containers never need AWS credentials. Postgres data copied from your laptop is not included in the
directory copy; either transcribe again on the instance or dump and restore the database
(`pg_dump` / `pg_restore` against port 5434).

### gemini

- `.env`: `GOOGLE_API_KEY=...` (required). Install with the `gemini` extra (`pip install -e ".[gemini]"`).
- Recommended model: `gemini-3-flash-preview`.
- Notes: uses JSON mode with the response schema.

```yaml
llm:
  default: {provider: gemini, model: "gemini-3-flash-preview", temperature: 0.2}
```

### openai_compat (OpenAI, or a local vLLM / LM Studio / llama.cpp server)

- `.env` for OpenAI: `OPENAI_API_KEY=...` and `OPENAI_BASE_URL=https://api.openai.com/v1`. The key is
  required only for `api.openai.com`.
- `.env` for a local server: point `OPENAI_BASE_URL` at it and leave the key empty. Examples are
  `http://127.0.0.1:8000/v1` for vLLM and `http://127.0.0.1:1234/v1` for LM Studio.
- Model: the model name the server exposes.
- Notes: requests `json_schema` output and falls back to `json_object` if the server rejects that.

```yaml
llm:
  default: {provider: openai_compat, model: "<model name on the server>", temperature: 0.2}
```

### deepseek

- `.env`: `DEEPSEEK_API_KEY=...` (required). `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com`.
- Recommended model: `deepseek-chat`. `deepseek-reasoner` also works but is slower.
- Notes: the API is OpenAI-compatible but does not support `json_schema`. `aqg` therefore uses
  `response_format: json_object` and appends the JSON schema to the prompt.

```yaml
llm:
  default:      {provider: deepseek, model: "deepseek-chat", temperature: 0.2}
  s5_reference: {provider: deepseek, model: "deepseek-chat", temperature: 0.0}
```

### groq

- `.env`: `GROQ_API_KEY=...` (required, from console.groq.com). `GROQ_BASE_URL` defaults to
  `https://api.groq.com/openai/v1`.
- Recommended split (the shipped config): `openai/gpt-oss-120b` with `reasoning_effort: medium` for
  `s2_extract`, `s3_spec`, `s4_reconcile`, `s5_tests` and `s5_reference`, the steps that decide correctness.
  `openai/gpt-oss-20b` handles only triage and the statement prose. On the BIP324 talk, 20b did badly at the
  other steps: it found no concepts, left every fact unverified despite good sources, and failed to produce valid
  test plans. Groq limits each model separately, so 20b's share uses its own budget.
- When Groq's server-side JSON check rejects an answer (`400 json_validate_failed`), `aqg` treats it as an
  invalid answer and re-prompts through the normal repair loop (`max_schema_retries`), instead of failing. To see which models your key can use:
  `curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`.
- Notes: the API is OpenAI-compatible. `aqg` sends `json_schema`, and if a model rejects it with HTTP 400 it
  falls back to `json_object` with the schema in the prompt. Groq is very fast, but the free tier has
  a limit of 8000 tokens per minute per model, and one call uses 2–6k.
  - **Pacing (avoids 429s):** `aqg` reads Groq's `x-ratelimit-remaining-tokens` / `x-ratelimit-reset-tokens`
    headers per model. Before each call it estimates the tokens needed (prompt plus room for the answer) and,
    if they don't fit, waits only as long as the window needs to refill. The log shows this as `pacing to stay
    under the token rate limit`. If you upgrade your Groq tier, the higher limits come from the same headers, so
    nothing needs changing.
  - **Fallback on 429:** if a 429 still happens, `aqg` waits exactly as long as Groq says (up to 8 times per
    call, each at most 90 s). A longer wait means a daily quota, so the run fails with a clear message.
  - A concept that fails because of an error like a rate limit is recorded as `failed`, not `rejected`, so the
    next run tries it again.
- `reasoning_effort` (`low|medium|high`): gpt-oss reasoning tokens also count against the limit.

```yaml
llm:
  default:      {provider: groq, model: "openai/gpt-oss-20b",  temperature: 0.2, reasoning_effort: low}
  s2_extract:   {provider: groq, model: "openai/gpt-oss-120b", temperature: 0.1, reasoning_effort: medium}
  s3_spec:      {provider: groq, model: "openai/gpt-oss-120b", temperature: 0.1, reasoning_effort: medium}
  s5_reference: {provider: groq, model: "openai/gpt-oss-120b", temperature: 0.0, reasoning_effort: medium}
```
