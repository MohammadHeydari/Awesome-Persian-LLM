import time
import requests


TEST_CASES = [
    {
        "category": "simple_instruction",
        "prompt": "Write a short sentence about artificial intelligence in Persian.",
    },
    {
        "category": "reasoning",
        "prompt": "If a train travels at 120 km/h, how far does it go in 2.5 hours? Answer in Persian.",
    },
    {
        "category": "taarof",
        "prompt": "Explain what the Persian phrase 'ghaabel-e shomaa ro nadaare' means and when it is used.",
    },
    {
        "category": "summarization",
        "prompt": "Summarize in Persian: Artificial intelligence is a branch of computer science aimed at building systems that exhibit intelligent behavior.",
    },
    {
        "category": "coding",
        "prompt": "Write a Python function that takes a list and returns its average. Add a Persian docstring.",
    },
]


def evaluate_response(response: str) -> dict:
    """Basic quality check on model output."""
    has_persian = any("\u0600" <= c <= "\u06ff" for c in response)
    return {
        "has_persian": has_persian,
        "word_count": len(response.split()),
        "char_count": len(response),
    }


def test_ollama(model_name: str = "partai/dorna-llama3", base_url: str = "http://localhost:11434"):

    # Test a Persian LLM running locally via Ollama.


    print("\n" + "=" * 50)
    print(f"Ollama  |  model: {model_name}")
    print("=" * 50)

    results = []

    for case in TEST_CASES:
        print(f"\n[{case['category']}]  {case['prompt'][:70]}...")

        start = time.time()
        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "prompt": case["prompt"], "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "")
            elapsed = time.time() - start

            metrics = evaluate_response(answer)
            print(f"  response: {answer[:120]}...")
            print(f"  time: {elapsed:.2f}s  |  {metrics}")

            results.append({"category": case["category"], "success": True, "time": elapsed, **metrics})

        except requests.exceptions.ConnectionError:
            print("  ERROR: Ollama is not running. Start it with: ollama serve")
            results.append({"category": case["category"], "success": False})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"category": case["category"], "success": False})

    return results



def test_huggingface(model_id: str = "universitytehran/PersianMind-v1.0"):

    # Load and test a Persian model directly from HuggingFace Hub.

    print("\n" + "=" * 50)
    print(f"HuggingFace  |  model: {model_id}")
    print("=" * 50)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print("  Loading model weights (this may take a few minutes)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        prompt = TEST_CASES[0]["prompt"]
        print(f"\n  prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        elapsed = time.time() - start

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer[len(prompt):].strip()

        metrics = evaluate_response(answer)
        print(f"  response: {answer[:150]}")
        print(f"  time: {elapsed:.2f}s  |  {metrics}")

        return {"success": True, "time": elapsed, **metrics}

    except ImportError:
        print("  ERROR: run `pip install transformers torch`")
        return {"success": False}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"success": False}


if __name__ == "__main__":
    print("Persian LLM Quick Start Test")
    print("=" * 50)

    # 1. Local model via Ollama
    #    requires: ollama pull partai/dorna-llama3 && ollama serve
    test_ollama(model_name="partai/dorna-llama3")

    # 2. HuggingFace
    #    test_huggingface(model_id="universitytehran/PersianMind-v1.0")

    print("\nDone.")