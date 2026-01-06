# A²-RAG Quick Start Guide

Get up and running with A²-RAG in 5 minutes ⚡

---

## Step 1: Clone & Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/saikiranpulagalla/A2_RAG_QA.git
cd A2_RAG_QA

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Configure API Keys (1 minute)

Create a `.env` file in the project root:

```bash
# Option A: Using terminal
echo "OPENROUTER_API_KEY=sk_your_key_here" > .env
echo "OPENAI_API_KEY=sk_your_key_here" >> .env

# Option B: Create .env manually
# Copy from .env.example and fill in your keys
cp .env.example .env
# Edit .env with your API keys
```

**Where to get keys:**
- **OpenRouter**: https://openrouter.ai/keys (free tier available)
- **OpenAI**: https://platform.openai.com/api-keys

---

## Step 3: Run a Quick Test (1 minute)

### Test A: Single Question

```python
from a2_rag.a2_pipeline import A2RAG
from data import load_documents

# Load documents
documents = load_documents("data/documents/wiki_docs.json")

# Create A²-RAG model
model = A2RAG(documents)

# Ask a question
query = "What is the capital of France?"
answer = model.answer(query)

print(f"Q: {query}")
print(f"A: {answer}")
```

### Test B: Full Evaluation (1 minute)

```bash
python example_usage.py
```

This runs:
- Both Baseline and A²-RAG on 50 test questions
- Generates comparison metrics
- Saves results to `results/` folder

**Expected output:**
```
Baseline RAG: F1=0.5787, EM=0.3800
A²-RAG: F1=0.5185, EM=0.3400
Results saved to results/comparison.csv
```

---

## Step 4: View Results (30 seconds)

Results are saved in `results/`:

```bash
# View summary
cat results/summary.json

# View comparison
cat results/comparison.csv

# View detailed results
cat results/a2rag_per_question.csv
cat results/baseline_per_question.csv
```

**Key metrics:**
- **F1 Score**: Quality of answers (0-1, higher is better)
- **Latency**: Response time in seconds
- **API Calls**: Number of LLM queries made
- **Hit Rate**: Percentage of relevant documents retrieved

---

## Step 5: Explore the Code (Optional)

### Core Files to Understand

```
a2_rag/
├── a2_pipeline.py           # Main orchestrator
├── agent_decision.py        # Decides if retrieval is needed
├── parent_child_retrieval.py # Two-stage retrieval
└── late_chunking.py         # Handles document chunking

baseline_rag/
└── baseline_pipeline.py     # Always retrieves documents

evaluation/
└── evaluate.py              # Metrics & comparison
```

### Quick Example: Custom Evaluation

```python
from a2_rag.a2_pipeline import A2RAG
from baseline_rag.baseline_pipeline import BaselineRAG
from evaluation.evaluate import compare_models
from data import load_documents, load_questions

# Load data
documents = load_documents("data/documents/wiki_docs.json")
questions = load_questions("data/questions/nq_1000.json")

# Initialize models
baseline = BaselineRAG(documents)
a2rag = A2RAG(documents)

# Evaluate on first 10 questions
results = compare_models(
    {"Baseline": baseline, "A²-RAG": a2rag},
    questions[:10]
)

# Print results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  EM: {metrics['em']:.4f}")
    print(f"  Latency: {metrics['latency']:.4f}s")
    print(f"  API Calls: {metrics['api_calls']:.2f}")
```

---

## Configuration Tips

Edit `config.py` to customize behavior:

```python
# Controls when retrieval is triggered (0-1 scale)
RETRIEVAL_DECISION_CONFIDENCE_THRESHOLD = 0.35  # Lower = more retrievals

# Retrieval parameters
PARENT_K = 3    # Top documents to retrieve
CHILD_K = 3     # Top chunks within documents

# Models
DECISION_LLM_MODEL = "openai/gpt-3.5-turbo"  # Via OpenRouter
LLM_MODEL = "openai/gpt-3.5-turbo"           # Via OpenRouter
EMBEDDING_MODEL = "text-embedding-3-small"   # OpenAI

# Chunk size for late chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

---

## Troubleshooting

### ❌ "API Key not found"
```bash
# Check .env file exists
ls -la .env

# Should contain:
# OPENROUTER_API_KEY=sk_...
# OPENAI_API_KEY=sk_...

# Verify keys are set
echo $OPENROUTER_API_KEY  # Should print your key
```

### ❌ "FAISS not installed"
```bash
pip install faiss-cpu
# or for GPU
pip install faiss-gpu
```

### ❌ "ModuleNotFoundError: No module named 'a2_rag'"
```bash
# Ensure you're in project root
cd A2_RAG_QA

# Verify directory structure
ls a2_rag/
ls baseline_rag/
```

### ❌ "Slow responses / timeout"
- Check internet connection
- Verify API rate limits
- Use smaller dataset: `questions[:5]` instead of full set
- Check `config.py` MAX_RETRIES and timeouts

---

## What's Next?

After the quick start, explore:

1. **Read the README** for complete documentation
   ```bash
   cat README.md
   ```

2. **Review design decisions** in README section "Design Decisions"
   - Why agentic decision?
   - Why parent-child retrieval?
   - Why OpenRouter?

3. **Examine evaluation results** in `results/` folder
   - Compare metrics between Baseline and A²-RAG
   - Check visualizations (5 PNG files)
   - Review per-question results

4. **Tune configuration** in `config.py`
   - Adjust retrieval threshold
   - Change chunk sizes
   - Modify TOP_K parameters

5. **Read the case study** for implementation insights
   ```bash
   open A2-RAG_Case_Study.docx
   ```

---

## Key Concepts

**Baseline RAG**
- Always retrieves documents
- Uses early chunking (chunks entire corpus upfront)
- Simple but potentially wasteful

**A²-RAG** (Proposed)
- Agent decides if retrieval is needed
- Uses parent-child hierarchical retrieval
- Late chunking (chunks only retrieved documents)
- More efficient but requires tuning

**Evaluation Metrics**
- **F1**: Token-level overlap (0-1)
- **EM**: Exact match (0 or 1)
- **Hit Rate**: Relevant docs retrieved (%)
- **Latency**: Time per query (seconds)
- **API Calls**: Number of LLM invocations

---

## File Structure at a Glance

```
A2-RAG-QA/
├── config.py                    # Configuration (edit this!)
├── example_usage.py             # Full pipeline example
├── requirements.txt             # Dependencies
├── .env                         # Your API keys (keep secret!)
│
├── a2_rag/                      # A²-RAG system
│   ├── a2_pipeline.py
│   ├── agent_decision.py
│   ├── parent_child_retrieval.py
│   └── late_chunking.py
│
├── baseline_rag/                # Baseline system
│   └── baseline_pipeline.py
│
├── evaluation/                  # Metrics
│   └── evaluate.py
│
├── embeddings/                  # Embeddings
│   └── embedder.py
│
├── vectorstore/                 # Vector storage
│   └── faiss_store.py
│
├── data/                        # Datasets
│   ├── documents/wiki_docs.json
│   └── questions/nq_1000.json
│
├── results/                     # Results (generated)
│   ├── comparison.csv
│   ├── summary.json
│   └── *.png (visualizations)
│
├── README.md                    # Full documentation
├── QUICK_START.md              # This file
├── A2-RAG_Case_Study.docx      # Case study
└── A2-RAG_Comprehensive_Research_Document.docx
```

---

## Common Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run evaluation
python example_usage.py

# Test single question
python -c "
from a2_rag.a2_pipeline import A2RAG
from data import load_documents
docs = load_documents('data/documents/wiki_docs.json')
model = A2RAG(docs)
print(model.answer('What is AI?'))
"

# View results
cat results/summary.json
cat results/comparison.csv

# Check configuration
grep -E "THRESHOLD|PARENT_K|CHILD_K" config.py
```

---

## Need Help?

1. Check the full [README.md](README.md)
2. Review [A2-RAG_Case_Study.docx](A2-RAG_Case_Study.docx)
3. Look at [example_usage.py](example_usage.py) for code examples
4. Check [config.py](config.py) for all available settings

---

**Ready to explore?** Run `python example_usage.py` now! 🚀
